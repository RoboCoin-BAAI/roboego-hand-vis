"""Write approved common-task mapping only. No backups; preserve task_name exactly."""
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


def apply_one(row,roots):
    root=roots[row['dataset']];path=root/row['path']
    if path.is_symlink() or root not in path.resolve().parents:raise ValueError('Unsafe target')
    raw=path.read_bytes();data=json.loads(raw.decode('utf-8-sig'))
    if data.get('task_name')!=row['task_name']:raise ValueError('Fine task_name changed since approved mapping')
    name=row['common_task_name'];changed=data.get('common_task_name')!=name
    revised=patch_common(raw,name) if changed else raw
    if changed:
        if path.read_bytes()!=raw:raise ValueError('Concurrent annotation modification')
        atomic_write(path,revised,stat.S_IMODE(path.stat().st_mode))
    actual=path.read_bytes()
    if actual!=revised or json.loads(actual.decode('utf-8-sig'))!=dict(data,common_task_name=name):
        raise ValueError('Common-task-only write verification failed')
    return dict(dataset=row['dataset'],path=row['path'],status='done',changed=changed,
                task_name=data['task_name'],common_task_name=name,verified=True,
                sha256=hashlib.sha256(actual).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--base',type=Path,default=Path('/share/project/liuyou/humancoin/data'))
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args()
    roots={'jianzhi':(a.base/'jianzhi/wait_for_publish').resolve(),'guanglun':(a.base/'guanglun_depth').resolve()}
    report=a.report_dir.resolve()
    if any(report==root or root in report.parents for root in roots.values()):raise ValueError('Reports must be outside datasets')
    if a.workers<1:raise ValueError('Invalid worker count')
    rows=list(csv.DictReader(a.selection.open(encoding='utf-8-sig')))
    if Counter(r['dataset'] for r in rows)!=Counter(jianzhi=7078,guanglun=9181):raise ValueError('Unexpected sample scope')
    if len({(r['dataset'],r['path']) for r in rows})!=len(rows):raise ValueError('Duplicate samples')
    for r in rows:
        match=re.fullmatch(r'batch_(\d{7})/[a-f0-9]+/annotation/episode_000000_structured\.json',r['path'])
        if not match or not 1<=int(match[1])<=({'jianzhi':71,'guanglun':100}[r['dataset']]):raise ValueError('Unsafe path/range')
        if not r['task_name'] or not re.fullmatch('[a-z][a-z0-9_]*',r['common_task_name']):raise ValueError('Invalid task name')
    counts={ds:len({r['common_task_name'] for r in rows if r['dataset']==ds}) for ds in roots}
    if counts!={'jianzhi':401,'guanglun':680} or len({r['common_task_name'] for r in rows})!=903:
        raise ValueError('Expected the approved 903-class mapping')
    report.mkdir(parents=True,exist_ok=True);results=[]
    with (report/'progress.jsonl').open('a',encoding='utf-8') as log,ThreadPoolExecutor(a.workers) as pool:
        jobs={pool.submit(apply_one,r,roots):r for r in rows}
        for i,future in enumerate(as_completed(jobs),1):
            src=jobs[future]
            try:r=future.result()
            except Exception as exc:r=dict(dataset=src['dataset'],path=src['path'],status='error',error=str(exc))
            results.append(r);log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            if i%500==0 or i==len(rows):print(json.dumps(dict(completed=i,total=len(rows),errors=sum(r['status']=='error' for r in results))),flush=True)
    summary=dict(selected=len(rows),changed=sum(r.get('changed',False) for r in results),
                 unchanged=sum(r['status']=='done' and not r['changed'] for r in results),
                 errors=sum(r['status']=='error' for r in results),scope=['common_task_name'],backup_created=False,
                 common_types=len({r['common_task_name'] for r in results if r['status']=='done'}),
                 datasets={ds:dict(selected=sum(r['dataset']==ds for r in results),
                                   changed=sum(r.get('changed',False) for r in results if r['dataset']==ds),
                                   errors=sum(r['status']=='error' for r in results if r['dataset']==ds),
                                   common_types=len({r['common_task_name'] for r in results if r['dataset']==ds and r['status']=='done'})) for ds in roots})
    save_json(report/'apply_report.json',dict(summary=summary,records=results));save_json(report/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)


if __name__=='__main__':raise SystemExit(main())
