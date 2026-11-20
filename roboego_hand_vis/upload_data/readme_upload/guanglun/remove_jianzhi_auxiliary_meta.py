"""Delete only the two approved auxiliary meta files and update readme sizes."""
import hashlib
import json
import stat
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from apply_scene_classification import atomic_write
from reclassify_scenes_qwen import save_json

BASE=Path('/share/project/liuyou/humancoin/data')
ROOT=BASE/'jianzhi/wait_for_publish/publish'
REPORT=BASE/'jianzhi_remove_auxiliary_meta_20261119'
NAMES=('alignment_check.jsonl','annotations.json')

def prepare(r):
    session=Path(r['target'])
    if ROOT not in session.parents:raise ValueError('Unsafe source')
    readme=session/'meta/readme.json';raw=readme.read_bytes();d=json.loads(raw.decode('utf-8-sig'))
    files=[]
    for name in NAMES:
        p=session/'meta'/name
        if p.is_symlink():raise ValueError('Symlink auxiliary file')
        if p.exists():
            if not p.is_file():raise ValueError('Not a file')
            s=p.stat();files.append(dict(name=name,size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino))
    return dict(session=str(session),files=files,readme_sha256=hashlib.sha256(raw).hexdigest(),old_count=d['file_count'],old_size=d['size_bytes'])

def remove(r):
    result=dict(session=r['session'],status='done',removed=[])
    try:
        session=Path(r['session']);rp=session/'meta/readme.json';raw=rp.read_bytes();d=json.loads(raw.decode('utf-8-sig'))
        expected_count=r['old_count']-len(r['files']);expected_size=r['old_size']-sum(f['size'] for f in r['files'])
        if not r['files']:
            result['file_count']=d['file_count'];return result
        if d['file_count']==expected_count and d['size_bytes']==expected_size and not any((session/'meta'/f['name']).exists() for f in r['files']):
            result.update(resumed=True,file_count=expected_count);return result
        if hashlib.sha256(raw).hexdigest()!=r['readme_sha256']:raise ValueError('Concurrent readme change')
        for f in r['files']:
            if f['name'] not in NAMES:raise ValueError('Unsafe auxiliary name')
            p=session/'meta'/f['name']
            if not p.exists():continue
            s=p.stat()
            if p.is_symlink() or (s.st_size,s.st_mtime_ns,s.st_ino)!=(f['size'],f['mtime_ns'],f['inode']):raise ValueError('Auxiliary file changed')
        revised=dict(d,file_count=expected_count,size_bytes=expected_size)
        payload=(json.dumps(revised,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        for f in r['files']:
            p=session/'meta'/f['name']
            if p.exists():p.unlink();result['removed'].append(f['name'])
        atomic_write(rp,payload,stat.S_IMODE(rp.stat().st_mode))
        if rp.read_bytes()!=payload or any((session/'meta'/name).exists() for name in NAMES):raise ValueError('Verification failed')
        result['file_count']=expected_count
    except Exception as exc:result.update(status='error',error=str(exc))
    return result

def main():
    REPORT.mkdir(parents=True,exist_ok=True);pp=REPORT/'plan.json'
    if pp.exists():plan=json.loads(pp.read_text())['records']
    else:
        rows=json.loads((BASE/'jianzhi_publish_dataset_move_20261119/plan.json').read_text())['records']
        assert len(rows)==7078
        with ThreadPoolExecutor(16) as pool:plan=list(pool.map(prepare,rows))
        save_json(pp,dict(records=plan))
    results=[]
    with (REPORT/'progress.jsonl').open('a') as log,ThreadPoolExecutor(16) as pool:
        for f in as_completed([pool.submit(remove,r) for r in plan]):
            r=f.result();results.append(r);log.write(json.dumps(r)+'\n');log.flush()
            if len(results)%1000==0:print(json.dumps(dict(completed=len(results),total=len(plan),errors=sum(r['status']=='error' for r in results))),flush=True)
    summary=dict(samples=len(results),errors=sum(r['status']=='error' for r in results),removed_files=dict(Counter(name for r in results for name in r['removed'])),removed_bytes=sum(f['size'] for r in plan for f in r['files']),readme_count_distribution=dict(Counter(r.get('file_count') for r in results if r['status']=='done')),scope=['meta/alignment_check.jsonl','meta/annotations.json','meta/readme.json:file_count,size_bytes'],backup_created=False)
    save_json(REPORT/'apply_report.json',dict(summary=summary,records=results));save_json(REPORT/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)

if __name__=='__main__':raise SystemExit(main())
