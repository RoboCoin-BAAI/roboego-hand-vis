"""Sync only info/episodes/tasks metadata to structured annotation task names."""
import copy
import json
import stat
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from apply_scene_classification import atomic_write
from reclassify_scenes_qwen import save_json

BASE=Path('/share/project/liuyou/humancoin/data')
ROOTS={'jianzhi':BASE/'jianzhi/wait_for_publish','guanglun':BASE/'guanglun_depth'}
REPORT=BASE/'meta_task_names_sync_20261119'

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def sync(row):
    result=dict(dataset=row['dataset'],path=row['path'],status='done',changed_files=[])
    try:
        root=ROOTS[row['dataset']].resolve();annotation_path=root/row['path'];session=annotation_path.parent.parent
        if root not in session.resolve().parents or session.is_symlink():raise ValueError('Unsafe path')
        annotations=sorted((session/'annotation').glob('episode_*_structured.json'))
        if annotations!=[annotation_path]:raise ValueError('Expected one structured episode')
        annotation_raw=annotation_path.read_bytes();annotation=json.loads(annotation_raw.decode('utf-8-sig'))
        name=annotation['task_name']
        if not isinstance(name,str) or not name.strip():raise ValueError('Missing task_name')
        meta=session/'meta';ip=meta/'info.json';ep=meta/'episodes.jsonl';tp=meta/'tasks.jsonl'
        info=json.loads(ip.read_text(encoding='utf-8-sig'));episodes=read_jsonl(ep);tasks=read_jsonl(tp)
        if len(episodes)!=1 or episodes[0]['episode_index']!=0 or len(tasks)!=1:raise ValueError('Expected single episode and task')
        if info.get('total_episodes')!=1:raise ValueError('Unexpected episode count')
        if not isinstance(tasks[0].get('task_index'),int):raise ValueError('Missing task_index')
        old_info=copy.deepcopy(info);old_episodes=copy.deepcopy(episodes);old_tasks=copy.deepcopy(tasks)
        tasks[0]['task']=name
        episodes[0]['tasks']=[name]
        info['total_tasks']=1
        if 'task_name' in info:info['task_name']=name
        staged=[]
        for path,old,new,is_lines in [(ip,old_info,info,False),(ep,old_episodes,episodes,True),(tp,old_tasks,tasks,True)]:
            if old==new:continue
            raw=path.read_bytes()
            if path.is_symlink():raise ValueError('Symlink meta file')
            payload=(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in new) if is_lines else json.dumps(new,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
            staged.append((path,raw,payload))
        if annotation_path.read_bytes()!=annotation_raw:raise ValueError('Annotation changed concurrently')
        for path,raw,payload in staged:
            if path.read_bytes()!=raw:raise ValueError('Concurrent metadata change')
        for path,raw,payload in staged:
            atomic_write(path,payload,stat.S_IMODE(path.stat().st_mode))
            if path.read_bytes()!=payload:raise ValueError('Write verification failed')
            result['changed_files'].append(path.name)
        if read_jsonl(tp)!=tasks or read_jsonl(ep)!=episodes or json.loads(ip.read_text(encoding='utf-8-sig'))!=info:raise ValueError('Metadata verification failed')
        result.update(task_name=name,previous_task=old_tasks[0].get('task'),previous_episode_tasks=old_episodes[0].get('tasks'))
    except Exception as exc:result.update(status='error',error=str(exc))
    return result

def main():
    rows=json.loads((BASE/'common_verb_lt5_apply_20261119/apply_report.json').read_text())['records']
    assert len(rows)==16259 and len({(r['dataset'],r['path']) for r in rows})==16259
    REPORT.mkdir(parents=True,exist_ok=True);results=[]
    with (REPORT/'progress.jsonl').open('w') as log,ThreadPoolExecutor(16) as pool:
        for f in as_completed([pool.submit(sync,row) for row in rows]):
            r=f.result();results.append(r);log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            if len(results)%1000==0:print(json.dumps(dict(completed=len(results),total=len(rows),errors=sum(r['status']=='error' for r in results))),flush=True)
    summary=dict(samples=len(results),errors=sum(r['status']=='error' for r in results),changed=sum(bool(r['changed_files']) for r in results),scope=['info.json','episodes.jsonl','tasks.jsonl'],readme_modified=False,backup_created=False,datasets={ds:dict(samples=sum(r['dataset']==ds for r in results),changed=sum(bool(r['changed_files']) for r in results if r['dataset']==ds),errors=sum(r['status']=='error' for r in results if r['dataset']==ds),files={name:sum(name in r['changed_files'] for r in results if r['dataset']==ds) for name in ['info.json','episodes.jsonl','tasks.jsonl']}) for ds in ROOTS})
    save_json(REPORT/'apply_report.json',dict(summary=summary,records=results));save_json(REPORT/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)

if __name__=='__main__':raise SystemExit(main())
