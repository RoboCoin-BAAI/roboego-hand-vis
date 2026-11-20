"""Read complete subtask evidence for the unresolved sample preview. No writes to datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import json
from pathlib import Path
import re


def read_one(row,base):
    roots={'jianzhi':base/'jianzhi/wait_for_publish','guanglun':base/'guanglun_depth'}
    if row['dataset'] not in roots:raise ValueError('Unknown dataset')
    root=roots[row['dataset']].resolve();relative=Path(row['path'])
    if not re.fullmatch(r'batch_\d+/[a-f0-9]+/annotation/episode_000000_structured\.json',relative.as_posix()):raise ValueError('Unsafe path')
    path=root/relative
    if path.is_symlink() or root not in path.resolve().parents:raise ValueError('Unsafe target')
    raw=path.read_bytes();data=json.loads(raw.decode('utf-8-sig'))
    if data.get('task_name')!=row['old_task_name']:raise ValueError('Task name changed since preview')
    return dict(dataset=row['dataset'],path=row['path'],sha256=hashlib.sha256(raw).hexdigest(),
                old_task_name=data.get('task_name'),task_description=data.get('task_description'),
                original_description=data.get('original_description'),objects=data.get('objects',[]),
                actions=data.get('actions'),subtasks=data.get('subtask',data.get('subtasks',[])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path,default=Path('/share/project/liuyou/humancoin/data'));p.add_argument('--workers',type=int,default=16)
    args=p.parse_args()
    rows=[r for r in csv.DictReader(args.selection.open(encoding='utf-8-sig')) if r['status']=='review']
    if len({(r['dataset'],r['path']) for r in rows})!=len(rows):raise ValueError('Duplicate samples')
    result=[];errors=[]
    with ThreadPoolExecutor(args.workers) as pool:
        jobs={pool.submit(read_one,r,args.base):r for r in rows}
        for i,future in enumerate(as_completed(jobs),1):
            try:result.append(future.result())
            except Exception as exc:errors.append(dict(**jobs[future],error=str(exc)))
            if i%100==0:print(json.dumps(dict(completed=i,total=len(rows),errors=len(errors))),flush=True)
    result.sort(key=lambda r:(r['dataset'],r['path']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise ValueError('Snapshot already exists; use a fresh output')
    args.output.write_text(json.dumps(dict(records=result,errors=errors),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected=len(rows),collected=len(result),errors=len(errors))),flush=True)


if __name__=='__main__':main()
