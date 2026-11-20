"""Move whole Jianzhi sample directories into dataset/unique publication layout."""
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from reclassify_scenes_qwen import save_json

BASE=Path('/share/project/liuyou/humancoin/data')
ROOT=(BASE/'jianzhi/wait_for_publish').resolve()
DEST=ROOT/'publish'
REPORT=BASE/'jianzhi_publish_dataset_move_20261119'
OLD_TARGETS={r['source']:r['target'] for r in json.loads((BASE/'jianzhi_publish_move_20261119/plan.json').read_text())['records']}

def inspect(row):
    relative=Path(row['path']).parent.parent
    if not re.fullmatch(r'batch_\d{7}/[a-f0-9]+',relative.as_posix()):raise ValueError('Unsafe source')
    source=ROOT/relative
    original_source=str(source)
    if not source.exists():source=Path(OLD_TARGETS[original_source])
    if source.is_symlink() or source.resolve()!=source:raise ValueError('Symlink source')
    readme=json.loads((source/'meta/readme.json').read_text(encoding='utf-8-sig'))
    annotation=json.loads((source/'annotation'/Path(row['path']).name).read_text(encoding='utf-8-sig'))
    common=readme['common_task_name'];unique=readme['unique_task_name'];dataset_name=readme['dataset_name']
    if common!=annotation['common_task_name'] or readme['task_name']!=annotation['task_name']:raise ValueError('Stale readme')
    area=(annotation.get('scene_classification') or {}).get('area') or (annotation.get('scene_classification') or {}).get('category') or 'unknown'
    expected=re.sub(r'[\W_]+','_','_'.join(['l1',area,annotation['task_name'],relative.name])).strip('_').lower()
    if unique!=expected:raise ValueError('Invalid unique_task_name')
    if not re.fullmatch('[a-z0-9_]+',dataset_name) or not re.fullmatch('[a-z0-9_]+',common) or not re.fullmatch('[a-z0-9_]+',unique):raise ValueError('Unsafe destination name')
    target=DEST/dataset_name/unique
    if target.exists() or target.is_symlink():raise ValueError('Destination collision: '+str(target))
    for name in ['meta','data','videos','annotation']:
        if not (source/name).is_dir():raise ValueError('Missing '+name)
    st=source.stat()
    if st.st_dev!=DEST.stat().st_dev:raise ValueError('Cross-filesystem move requires separate handling')
    return dict(original_source=original_source,dataset_name=dataset_name,source=str(source),target=str(target),common_task_name=common,unique_task_name=unique,inode=st.st_ino,device=st.st_dev)

def validate_plan(plan):
    if len(plan)!=7078 or len({r['source'] for r in plan})!=7078 or len({r['target'] for r in plan})!=7078:raise ValueError('Duplicate/missing samples')
    def check(r):
        source=Path(r['source']);target=Path(r['target'])
        if ROOT not in source.parents or DEST not in target.parents:raise ValueError('Unsafe plan')
        if target.parent.is_symlink() or target.parent.resolve()!=target.parent:raise ValueError('Unsafe target parent')
        if target.exists():
            if source.exists() or target.is_symlink():raise ValueError('Collision during resume')
            st=target.stat()
        else:
            if not source.is_dir() or source.is_symlink():raise ValueError('Missing source')
            st=source.stat()
        if st.st_ino!=r['inode'] or st.st_dev!=r['device']:raise ValueError('Directory identity changed')
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(check,plan))

def move_one(r):
    source=Path(r['source']);target=Path(r['target'])
    if target.exists():return dict(**r,status='done',resumed=True)
    target.parent.mkdir(exist_ok=True)
    if target.parent.is_symlink() or target.parent.resolve()!=target.parent:raise ValueError('Unsafe category directory')
    if target.exists() or target.is_symlink():raise ValueError('Destination appeared')
    # Rename on the same filesystem preserves all sample bytes and directory contents.
    os.rename(source,target)
    st=target.stat()
    if source.exists() or st.st_ino!=r['inode'] or st.st_dev!=r['device']:raise ValueError('Move verification failed')
    d=json.loads((target/'meta/readme.json').read_text(encoding='utf-8-sig'))
    if d['unique_task_name']!=r['unique_task_name'] or d['common_task_name']!=r['common_task_name']:raise ValueError('Moved labels mismatch')
    return dict(**r,status='done',resumed=False)

def main():
    if not DEST.is_dir() or DEST.is_symlink() or DEST.resolve()!=DEST:raise ValueError('Publish must be a real directory within root')
    REPORT.mkdir(parents=True,exist_ok=True);plan_path=REPORT/'plan.json'
    if plan_path.exists():plan=json.loads(plan_path.read_text())['records']
    else:
        rows=[r for r in json.loads((BASE/'common_verb_lt5_apply_20261119/apply_report.json').read_text())['records'] if r['dataset']=='jianzhi']
        plan=[]
        with ThreadPoolExecutor(16) as pool:
            for future in as_completed([pool.submit(inspect,row) for row in rows]):
                plan.append(future.result())
                if len(plan)%1000==0:print(json.dumps(dict(phase='preflight',checked=len(plan),total=7078)),flush=True)
        save_json(plan_path,dict(records=sorted(plan,key=lambda r:r['source'])))
    validate_plan(plan)
    print(json.dumps(dict(phase='preflight_done',samples=len(plan),groups=len({r['dataset_name'] for r in plan}))),flush=True)
    results=[]
    with (REPORT/'progress.jsonl').open('a') as log,ThreadPoolExecutor(8) as pool:
        futures={pool.submit(move_one,r):r for r in plan}
        for future in as_completed(futures):
            try:r=future.result()
            except Exception as exc:r=dict(**futures[future],status='error',error=str(exc))
            results.append(r);log.write(json.dumps(r)+'\n');log.flush()
            if len(results)%500==0:print(json.dumps(dict(phase='move',completed=len(results),total=len(plan),errors=sum(r['status']=='error' for r in results))),flush=True)
    errors=[r for r in results if r['status']=='error']
    validate_plan(plan)
    summary=dict(samples=len(plan),moved=sum(r['status']=='done' for r in results),errors=len(errors),groups=len({r['dataset_name'] for r in plan}),destination=str(DEST),group_counts=dict(sorted(Counter(r['dataset_name'] for r in plan).items())),source_batch_directories_preserved=True)
    save_json(REPORT/'apply_report.json',dict(summary=summary,records=results));save_json(REPORT/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
