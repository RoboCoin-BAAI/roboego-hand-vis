"""Collect live task names, without modifying annotations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import json
from pathlib import Path


def batch_rows(dataset, root, number):
    batch=root/f'batch_{number:07d}'
    if not batch.is_dir():return [],[dict(dataset=dataset,path=str(batch),error='Missing batch')]
    rows=[];errors=[]
    for sample in sorted(batch.iterdir()):
        if not sample.is_dir() or sample.name.startswith(('.', '_')) or sample.name=='logs':continue
        path=sample/'annotation/episode_000000_structured.json'
        try:
            if path.is_symlink() or root not in path.resolve().parents:raise ValueError('Unsafe path')
            d=json.loads(path.read_text(encoding='utf-8-sig'))
            rows.append(dict(dataset=dataset,path=path.relative_to(root).as_posix(),task_name=d.get('task_name'),
                             common_task_name=d.get('common_task_name'),objects=[o.get('name') for o in d.get('objects',[])],
                             task_description=d.get('task_description')))
        except Exception as exc:errors.append(dict(dataset=dataset,path=str(path),error=str(exc)))
    return rows,errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('/share/project/liuyou/humancoin/data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();rows=[];errors=[]
    with ThreadPoolExecutor(a.workers) as pool:
        jobs=[pool.submit(batch_rows,ds,root.resolve(),n) for ds,root,end in
              [('jianzhi',a.base/'jianzhi/wait_for_publish',71),('guanglun',a.base/'guanglun_depth',100)]
              for n in range(1,end+1)]
        for i,f in enumerate(as_completed(jobs),1):
            r,e=f.result();rows.extend(r);errors.extend(e)
            if i%10==0:print(json.dumps(dict(batches=i,total=171,samples=len(rows),errors=len(errors))),flush=True)
    rows.sort(key=lambda r:(r['dataset'],r['path']))
    summary={}
    for ds in ('jianzhi','guanglun'):
        selected=[r for r in rows if r['dataset']==ds]
        names=Counter(r['task_name'] for r in selected if isinstance(r['task_name'],str) and r['task_name'].strip())
        summary[ds]=dict(samples=len(selected),unique=len(names),missing=len(selected)-sum(names.values()),names=dict(names.most_common()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(summary=summary,errors=errors,records=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({ds:{k:v for k,v in s.items() if k!='names'} for ds,s in summary.items()}),flush=True)


if __name__=='__main__':main()
