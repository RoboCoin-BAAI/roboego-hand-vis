"""Set common_task_name to <fine-task verb>_task, no object classes/backups."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import json
from pathlib import Path
import re
import stat

from apply_scene_classification import atomic_write
from classify_common_tasks_qwen import patch_common
from reclassify_scenes_qwen import save_json


def common_name(task,merges=None,context='',round2=False):
    if not isinstance(task,str) or not task.strip():raise ValueError('Missing fine task_name')
    task=' '.join(task.lower().split())
    if task=='origanize_some_object':return 'organize_task'
    verb=task.split(' ',1)[0].split('_',1)[0]
    if not re.fullmatch('[a-z]+',verb):raise ValueError('Invalid primary verb: '+task)
    if round2:
        ctx=context.lower();noun=task.partition(' ')[2]
        if verb=='collect' and (re.search(r'\b(trash|garbage|waste|debris|scraps?|clippings?)\b',noun) or
             re.search(r'\b(paper|hair)\b',noun) and re.search(r'\b(trash can|trash bin|garbage|clippings|shredded|discard|throw)\b',ctx)):
            return 'clean_task'
        if verb=='hang':
            if re.search(r'\b(dry|drying|air.dry)\b',ctx):return 'dry_task'
            if re.search(r'\b(install|mount|mounting|ceiling|fixture|lamp)\b',ctx):return 'hang_task'
        if verb in ('insert','plug','connect') and re.search(r'\b(charg\w*|power socket|power outlet|electrical outlet)\b',ctx):return verb+'_task'
        if verb=='screw' and re.search(r'\b(unscrew|loosen)\b',ctx):return 'screw_task'
    return (merges or {}).get(verb,verb)+'_task'


def update(path,root,dataset,merges=None,round2=False):
    if path.is_symlink() or root not in path.resolve().parents:raise ValueError('Unsafe target')
    raw=path.read_bytes();data=json.loads(raw.decode('utf-8-sig'))
    context=' '.join(str(data.get(k) or '') for k in ('task_description','original_description'))
    name=common_name(data.get('task_name'),merges,context,round2)
    changed=data.get('common_task_name')!=name
    revised=patch_common(raw,name) if changed else raw
    if changed:
        if path.read_bytes()!=raw:raise ValueError('Concurrent modification')
        atomic_write(path,revised,stat.S_IMODE(path.stat().st_mode))
    actual=path.read_bytes()
    if actual!=revised or json.loads(actual.decode('utf-8-sig'))!=dict(data,common_task_name=name):
        raise ValueError('Unexpected modification outside common_task_name')
    return dict(dataset=dataset,path=path.relative_to(root).as_posix(),status='done',changed=changed,
                task_name=data['task_name'],common_task_name=name,previous_common_task_name=data.get('common_task_name'),
                byte_delta=len(actual)-len(raw),verified=True,sha256=hashlib.sha256(actual).hexdigest())


def session_job(row,roots,merges=None,round2=False):
    root=roots[row['dataset']];primary=root/row['path'];results=[]
    paths=sorted(primary.parent.glob('episode_*_structured.json'))
    if primary not in paths:raise ValueError('Expected primary annotation missing')
    for path in paths:
        try:results.append(update(path,root,row['dataset'],merges,round2))
        except Exception as exc:results.append(dict(dataset=row['dataset'],path=path.relative_to(root).as_posix(),status='error',error=str(exc)))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--base',type=Path,default=Path('/share/project/liuyou/humancoin/data'))
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--merge-rules',type=Path)
    p.add_argument('--round2-context',action='store_true')
    args=p.parse_args()
    roots={'jianzhi':(args.base/'jianzhi/wait_for_publish').resolve(),'guanglun':(args.base/'guanglun_depth').resolve()}
    if args.workers<1:raise ValueError('Invalid workers')
    report=args.report_dir.resolve()
    if any(report==root or root in report.parents for root in roots.values()):raise ValueError('Report must be outside datasets')
    rows=list(csv.DictReader(args.selection.open(encoding='utf-8-sig')))
    if Counter(r['dataset'] for r in rows)!=Counter(jianzhi=7078,guanglun=9181):raise ValueError('Unexpected scope')
    if len({(r['dataset'],r['path']) for r in rows})!=len(rows):raise ValueError('Duplicate sessions')
    for r in rows:
        m=re.fullmatch(r'batch_(\d{7})/[a-f0-9]+/annotation/episode_000000_structured\.json',r['path'])
        if not m or not 1<=int(m[1])<=({'jianzhi':71,'guanglun':100}[r['dataset']]):raise ValueError('Unsafe path')
    merges=json.loads(args.merge_rules.read_text(encoding='utf-8')) if args.merge_rules else {}
    if not isinstance(merges,dict) or any(not re.fullmatch('[a-z]+',k) or not isinstance(v,str) or not re.fullmatch('[a-z]+',v) for k,v in merges.items()):raise ValueError('Invalid verb merge rules')
    report.mkdir(parents=True,exist_ok=True);results=[]
    save_json(report/'effective_verb_merges.json',merges)
    with (report/'progress.jsonl').open('a',encoding='utf-8') as log,ThreadPoolExecutor(args.workers) as pool:
        jobs={pool.submit(session_job,r,roots,merges,args.round2_context):r for r in rows}
        for i,f in enumerate(as_completed(jobs),1):
            try:events=f.result()
            except Exception as exc:events=[dict(dataset=jobs[f]['dataset'],path=jobs[f]['path'],status='error',error=str(exc))]
            for r in events:results.append(r);log.write(json.dumps(r,ensure_ascii=False)+'\n')
            log.flush()
            if i%500==0 or i==len(rows):print(json.dumps(dict(completed_sessions=i,total_sessions=len(rows),annotations=len(results),errors=sum(r['status']=='error' for r in results))),flush=True)
    if len({(r['dataset'],r['path']) for r in results})!=len(results):raise ValueError('Duplicate output annotations')
    summary=dict(sessions=len(rows),annotations=len(results),changed=sum(r.get('changed',False) for r in results),
                 errors=sum(r['status']=='error' for r in results),backup_created=False,scope=['common_task_name'],
                 common_types=len({r['common_task_name'] for r in results if r['status']=='done'}),
                 datasets={ds:dict(annotations=sum(r['dataset']==ds for r in results),changed=sum(r.get('changed',False) for r in results if r['dataset']==ds),
                    errors=sum(r['status']=='error' for r in results if r['dataset']==ds),
                    common_types=len({r['common_task_name'] for r in results if r['dataset']==ds and r['status']=='done'})) for ds in roots})
    save_json(report/'apply_report.json',dict(summary=summary,records=results));save_json(report/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)


if __name__=='__main__':raise SystemExit(main())
