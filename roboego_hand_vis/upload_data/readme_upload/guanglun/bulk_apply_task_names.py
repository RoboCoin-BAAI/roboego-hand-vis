"""Apply approved fine task-name preview, WITHOUT backups. Only task_name changes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import re
import stat

from apply_scene_classification import atomic_write, object_value_spans
from reclassify_scenes_qwen import save_json


def patch(raw,name):
    text=raw.decode('utf-8-sig');data=json.loads(text);spans=object_value_spans(text)
    if 'task_name' in spans:
        first,last=spans['task_name'];updated=text[:first]+json.dumps(name)+text[last:]
    else:
        end=max((v[1] for v in spans.values()),default=text.index('{')+1)
        updated=text[:end]+(',' if spans else '')+'\n  "task_name": '+json.dumps(name)+text[end:]
    if json.loads(updated)!=dict(data,task_name=name):raise ValueError('Unexpected modification outside task_name')
    return (b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b'')+updated.encode('utf-8')


def apply_one(row,roots):
    root=roots[row['dataset']];relative=Path(row['path']);path=root/relative
    if path.is_symlink() or root not in path.resolve().parents:raise ValueError('Unsafe target')
    raw=path.read_bytes();data=json.loads(raw.decode('utf-8-sig'));name=row['proposed']
    before=data.get('task_name');changed=before!=name
    if changed:
        expected=row['current_known_task_name'] or None
        if before!=expected:raise ValueError('Current task_name differs from approved source: '+repr(before))
        revised=patch(raw,name)
        if path.read_bytes()!=raw:raise ValueError('Concurrent modification')
        atomic_write(path,revised,stat.S_IMODE(path.stat().st_mode))
    else:revised=raw
    actual=path.read_bytes()
    if actual!=revised or json.loads(actual.decode('utf-8-sig'))!=dict(data,task_name=name):
        raise ValueError('Task-only write verification failed')
    return dict(dataset=row['dataset'],path=row['path'],status='done',changed=changed,
                task_name=name,verified=True,sha256=hashlib.sha256(actual).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--base',type=Path,default=Path('/share/project/liuyou/humancoin/data'))
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    args=p.parse_args()
    roots={'jianzhi':(args.base/'jianzhi/wait_for_publish').resolve(),'guanglun':(args.base/'guanglun_depth').resolve()}
    report=args.report_dir.resolve()
    if any(report==r or r in report.parents for r in roots.values()):raise ValueError('Report must be outside publication')
    rows=list(csv.DictReader(args.selection.open(encoding='utf-8-sig')))
    if Counter(r['dataset'] for r in rows)!=Counter(jianzhi=7078,guanglun=9181):raise ValueError('Unexpected scope')
    if len({(r['dataset'],r['path']) for r in rows})!=len(rows):raise ValueError('Duplicate path')
    for r in rows:
        match=re.fullmatch(r'batch_(\d{7})/[a-f0-9]+/annotation/episode_000000_structured\.json',r['path'])
        if not match or not 1<=int(match[1])<=({'jianzhi':71,'guanglun':100}[r['dataset']]):raise ValueError('Unsafe path/range')
        name=r['proposed']
        if not name or len(name)>100 or not re.fullmatch('[a-z][a-z _-]*',name):raise ValueError('Invalid task label')
    report.mkdir(parents=True,exist_ok=True)
    results=[]
    # No source-file copies/backups are created, per the user's explicit request.
    with (report/'progress.jsonl').open('a',encoding='utf-8') as log,ThreadPoolExecutor(args.workers) as pool:
        jobs={pool.submit(apply_one,r,roots):r for r in rows}
        for i,future in enumerate(as_completed(jobs),1):
            row=jobs[future]
            try:result=future.result()
            except Exception as exc:result=dict(dataset=row['dataset'],path=row['path'],status='error',error=str(exc))
            results.append(result);log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
            if i%500==0 or i==len(rows):print(json.dumps(dict(completed=i,total=len(rows),errors=sum(r['status']=='error' for r in results))),flush=True)
    summary=dict(selected=len(rows),changed=sum(r.get('changed',False) for r in results),
                 unchanged=sum(r['status']=='done' and not r['changed'] for r in results),
                 errors=sum(r['status']=='error' for r in results),backup_created=False,scope=['task_name'],
                 datasets={ds:dict(selected=sum(r['dataset']==ds for r in results),changed=sum(r.get('changed',False) for r in results if r['dataset']==ds),
                                   errors=sum(r['status']=='error' for r in results if r['dataset']==ds)) for ds in roots})
    save_json(report/'apply_report.json',dict(summary=summary,records=results))
    save_json(report/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)


if __name__=='__main__':raise SystemExit(main())
