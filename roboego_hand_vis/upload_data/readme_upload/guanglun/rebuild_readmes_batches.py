"""Run the existing L1 readme builder for approved batches and verify label fields."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time

from reclassify_scenes_qwen import save_json


def run_batch(key,rows,args):
    dataset,batch=key;root=args.base/('jianzhi/wait_for_publish' if dataset=='jianzhi' else 'guanglun_depth')
    log=args.report_dir/'logs'/f'{dataset}_{batch}.log';log.parent.mkdir(parents=True,exist_ok=True)
    command=[args.python,'-m','roboego_hand_vis.build_json.base.build_readme','--root',str(root/batch),'--level','L1']
    env=os.environ.copy();env['PATH']=str(Path(args.python).parent)+os.pathsep+env.get('PATH','')
    env.setdefault('OMP_NUM_THREADS','1');env.setdefault('OPENBLAS_NUM_THREADS','1')
    start=time.monotonic()
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=args.repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
    written=failed=0
    for line in log.read_text(encoding='utf-8',errors='replace').splitlines():
        if line.startswith('WROTE '):written+=1
        if line.startswith('FAILED '):failed+=1
    errors=[]
    if written!=len(rows):errors.append(f'Expected {len(rows)} readmes; wrote {written}')
    for row in rows:
        session=root/Path(row['path']).parent.parent
        try:
            readme=json.loads((session/'meta/readme.json').read_text(encoding='utf-8-sig'))
            annotation=json.loads((root/row['path']).read_text(encoding='utf-8-sig'))
            scene=annotation.get('scene_classification') or {}
            area=scene.get('area') or scene.get('category') or 'unknown'
            expected=re.sub(r'[\W_]+','_','_'.join(['l1',area,annotation.get('task_name') or 'unknown',session.name])).strip('_').lower()
            if readme.get('unique_task_name')!=expected:raise ValueError('unique_task_name mismatch')
            if readme.get('task_name')!=annotation.get('task_name'):raise ValueError('task_name mismatch')
            if readme.get('common_task_name')!=annotation.get('common_task_name'):raise ValueError('common_task_name mismatch')
            info=json.loads((session/'meta/info.json').read_text(encoding='utf-8-sig'))
            features=info['features'];main=features.get('observation.images.head_left_rectified')
            color_count=sum(f.get('dtype')=='video' and 'depth' not in key.lower() for key,f in features.items())
            expected_dataset=re.sub(r'[\W_]+','_','_'.join(['l1',area,annotation.get('common_task_name') or 'unknown',
                f"{main['shape'][1]}x{main['shape'][0]}" if main else 'unknown',
                f"{main.get('fps',info.get('fps')):g}fps" if main and main.get('fps',info.get('fps')) else 'unknown',f'{color_count}views'])).strip('_').lower()
            if readme.get('dataset_name')!=expected_dataset:raise ValueError('dataset_name mismatch')
            if readme.get('cameras',{}).get('color_count')!=color_count:raise ValueError('color_count mismatch')
            if 'common_task' in readme:raise ValueError('Obsolete common_task present')
            if {'quality_flags','eeg','emg','depth','glove'} & set(readme.get('modalities',{})):raise ValueError('Unexpected L1 modalities')
        except Exception as exc:errors.append(dict(path=row['path'],error=str(exc)))
    return dict(dataset=dataset,batch=batch,status='done' if not errors and not failed and result.returncode==0 else 'error',
                expected=len(rows),written=written,failed=failed,returncode=result.returncode,
                validation_errors=errors,seconds=round(time.monotonic()-start,2),log=str(log))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('/share/project/liuyou/humancoin/data'))
    p.add_argument('--repo',type=Path,default=Path('/share/project/liuyou/humancoin/code/roboego-hand-vis'))
    p.add_argument('--python',default='/share/project/liuyou/humancoin/envs/ego/bin/python')
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--jianzhi-end',type=int,default=71)
    p.add_argument('--guanglun-end',type=int,default=100)
    a=p.parse_args()
    if not 1<=a.jianzhi_end<=71 or not 1<=a.guanglun_end<=100 or a.workers<1:raise ValueError('Invalid range/workers')
    rows=list(csv.DictReader(a.mapping.open(encoding='utf-8-sig')))
    if Counter(r['dataset'] for r in rows)!=Counter(jianzhi=7078,guanglun=9181):raise ValueError('Unexpected scope')
    groups=defaultdict(list)
    for row in rows:
        match=re.fullmatch(r'(batch_(\d{7}))/[a-f0-9]+/annotation/episode_000000_structured\.json',row['path'])
        if not match or not 1<=int(match[2])<=({'jianzhi':71,'guanglun':100}[row['dataset']]):raise ValueError('Invalid batch path')
        if int(match[2])>({'jianzhi':a.jianzhi_end,'guanglun':a.guanglun_end}[row['dataset']]):continue
        groups[(row['dataset'],match[1])].append(row)
    expected_batches=a.jianzhi_end+a.guanglun_end
    if len(groups)!=expected_batches or len({(r['dataset'],r['path']) for r in rows})!=16259:raise ValueError('Duplicate/missing batches/samples')
    expected_samples=sum(len(value) for value in groups.values())
    a.report_dir.mkdir(parents=True,exist_ok=True);results=[];start=time.monotonic()
    with ThreadPoolExecutor(a.workers) as pool:
        futures={pool.submit(run_batch,key,value,a):key for key,value in sorted(groups.items())}
        pending=set(futures)
        while pending:
            done,pending=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for future in done:
                try:r=future.result()
                except Exception as exc:
                    ds,batch=futures[future];r=dict(dataset=ds,batch=batch,status='error',written=0,failed=0,validation_errors=[str(exc)])
                results.append(r)
                save_json(a.report_dir/'batch_results.json',dict(records=results))
            active_written=0
            for future in pending:
                if future.running():
                    ds,batch=futures[future];log=a.report_dir/'logs'/f'{ds}_{batch}.log'
                    if log.exists():active_written+=sum(line.startswith('WROTE ') for line in log.read_text(encoding='utf-8',errors='replace').splitlines())
            print(json.dumps(dict(completed_batches=len(results),total_batches=expected_batches,
                                  written=sum(r['written'] for r in results)+active_written,total_samples=expected_samples,
                                  error_batches=sum(r['status']=='error' for r in results),elapsed_s=round(time.monotonic()-start))),flush=True)
    summary=dict(batches=len(results),written=sum(r['written'] for r in results),failed=sum(r['failed'] for r in results),
                 ranges={'jianzhi':[1,a.jianzhi_end],'guanglun':[1,a.guanglun_end]},expected_samples=expected_samples,
                 error_batches=sum(r['status']=='error' for r in results),seconds=round(time.monotonic()-start,2),level='L1',
                 validation='task_name/common_task_name match current annotations; unique_task_name follows L1 naming; mapping selects paths only',
                 datasets={ds:dict(batches=sum(r['dataset']==ds for r in results),written=sum(r['written'] for r in results if r['dataset']==ds),
                     error_batches=sum(r['status']=='error' for r in results if r['dataset']==ds)) for ds in ('jianzhi','guanglun')})
    save_json(a.report_dir/'summary.json',summary);print(json.dumps(summary),flush=True)
    return int(summary['error_batches']>0 or summary['written']!=expected_samples)


if __name__=='__main__':raise SystemExit(main())
