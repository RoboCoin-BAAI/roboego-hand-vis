"""Remove approved inactive locks/logs/staging artifacts only from Guanglun publish."""
import fcntl
import hashlib
import json
import os
import stat
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import ExitStack
from pathlib import Path
import av
import pyarrow.parquet as pq
from apply_scene_classification import atomic_write
from reclassify_scenes_qwen import save_json

BASE=Path('/share/project/liuyou/humancoin/data')
ROOT=(BASE/'guanglun_depth/publish').resolve()
REPORT=BASE/'guanglun_publish_cleanup_20261119'
TEMP_NAMES={'depth_hevc12.ffmpeg.log','depth_hevc12.mp4','valid_mask_ffv1.mkv'}

def prepare(row):
    session=Path(row['source'])
    if ROOT not in session.parents or session.resolve()!=session:raise ValueError('Unsafe sample')
    files=[];dirs=[]
    for p in session.iterdir():
        if p.name.endswith('.publish.lock'):
            if p.is_symlink():raise ValueError('Symlink lock')
            with p.open('rb') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            files.append(p)
        elif p.name.startswith('.reencode-'):
            if not p.is_dir() or p.is_symlink() or p.resolve()!=p:raise ValueError('Unsafe temporary directory')
            children=list(p.iterdir())
            if {x.name for x in children}!=TEMP_NAMES or any(not x.is_file() or x.is_symlink() for x in children):raise ValueError('Unexpected staging contents')
            # Ensure the formal depth is complete and does not depend on this staging mask.
            info=json.loads((session/'meta/info.json').read_text())
            feature=info['features']['observation.images.depth_head_left_rectified']
            if feature.get('depth_storage',{}).get('requires_valid_mask'):raise ValueError('Separate valid mask required')
            if feature.get('info',{}).get('depth.method')!='da3_monocular':raise ValueError('Unexpected formal depth method')
            video=next((session/'videos').glob('chunk-*/observation.images.depth_head_left_rectified/episode_*.mp4'))
            table=next((session/'data').glob('chunk-*/episode_*/observation.images.depth_head_left_rectified.parquet'))
            with av.open(str(video)) as container:
                stream=container.streams.video[0]
                if stream.frames!=pq.ParquetFile(table).metadata.num_rows or stream.codec_context.name!='hevc':raise ValueError('Formal depth mismatch')
            files.extend(children);dirs.append(p)
    for p in (session/'videos').glob('chunk-*/*/logs/ks3util.log'):
        files.append(p);dirs.append(p.parent)
    records=[]
    for p in files:
        if p.is_symlink() or session not in p.resolve().parents or not p.is_file():raise ValueError('Unsafe cleanup file')
        st=p.stat();records.append(dict(path=str(p.relative_to(session)),size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino))
    readme=session/'meta/readme.json';raw=readme.read_bytes();d=json.loads(raw.decode('utf-8-sig'))
    if d['file_count']-len(records)!=29:raise ValueError('Unexpected files: '+str(session))
    return dict(session=str(session),files=records,directories=[str(p.relative_to(session)) for p in dirs],old_count=d['file_count'],old_size=d['size_bytes'],readme_sha256=hashlib.sha256(raw).hexdigest())

def cleanup(r):
    result=dict(session=r['session'],status='done',removed=[])
    try:
        session=Path(r['session']);rp=session/'meta/readme.json';raw=rp.read_bytes();d=json.loads(raw.decode('utf-8-sig'))
        if not r['files']:return result
        new_size=r['old_size']-sum(f['size'] for f in r['files'])
        if d['file_count']==29 and d['size_bytes']==new_size and all(not (session/f['path']).exists() for f in r['files']):
            result['resumed']=True;return result
        if hashlib.sha256(raw).hexdigest()!=r['readme_sha256']:raise ValueError('Concurrent readme modification')
        with ExitStack() as stack:
            for f in r['files']:
                p=session/f['path']
                if not p.exists():continue
                st=p.stat()
                if p.is_symlink() or (st.st_size,st.st_mtime_ns,st.st_ino)!=(f['size'],f['mtime_ns'],f['inode']):raise ValueError('Candidate changed')
                if p.name.endswith('.publish.lock'):
                    lock=stack.enter_context(p.open('rb'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for f in r['files']:
                p=session/f['path']
                if p.exists():p.unlink();result['removed'].append(f['path'])
            for relative in r['directories']:
                p=session/relative
                if p.exists():
                    if session not in p.resolve().parents or p.is_symlink():raise ValueError('Unsafe cleanup directory')
                    p.rmdir()
            revised=dict(d,file_count=29,size_bytes=new_size)
            payload=(json.dumps(revised,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
            atomic_write(rp,payload,stat.S_IMODE(rp.stat().st_mode))
            if rp.read_bytes()!=payload or any((session/f['path']).exists() for f in r['files']):raise ValueError('Cleanup verification failed')
    except Exception as exc:result.update(status='error',error=str(exc))
    return result

def main():
    REPORT.mkdir(exist_ok=True);pp=REPORT/'cleanup_plan.json'
    if pp.exists():plan=json.loads(pp.read_text())['records']
    else:
        rows=json.loads((BASE/'guanglun_badfps_move_01_49_20261119/selection.json').read_text())['retained']
        if len(rows)!=4338:raise ValueError('Wrong scope')
        with ThreadPoolExecutor(16) as pool:plan=list(pool.map(prepare,rows))
        save_json(pp,dict(records=plan))
    # Detect open file descriptors for the selected paths before any unlink.
    targets={str(Path(r['session'])/f['path']) for r in plan for f in r['files']}
    def open_targets(proc):
        found=[]
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    path=os.readlink(fd)
                    if path in targets:found.append(dict(pid=proc.name,path=path))
                except OSError:pass
        except OSError:pass
        return found
    with ThreadPoolExecutor(16) as pool:active=[x for xs in pool.map(open_targets,[p for p in Path('/proc').iterdir() if p.name.isdigit()]) for x in xs]
    if active:raise ValueError('Files still in use: '+json.dumps(active))
    print(json.dumps(dict(phase='preflight_done',samples=len(plan),files=len(targets))),flush=True)
    results=[]
    with (REPORT/'progress.jsonl').open('a') as log,ThreadPoolExecutor(16) as pool:
        for future in as_completed([pool.submit(cleanup,r) for r in plan]):
            r=future.result();results.append(r);log.write(json.dumps(r)+'\n');log.flush()
            if len(results)%1000==0:print(json.dumps(dict(completed=len(results),total=len(plan),errors=sum(r['status']=='error' for r in results))),flush=True)
    summary=dict(samples=len(results),errors=sum(r['status']=='error' for r in results),removed_files=sum(len(r['removed']) for r in results),removed_locks=sum(f['path'].endswith('.publish.lock') for r in plan for f in r['files']),removed_upload_logs=sum(f['path'].endswith('ks3util.log') for r in plan for f in r['files']),removed_temporary_directories=sum(Path(p).name.startswith('.reencode-') for r in plan for p in r['directories']),removed_bytes=sum(f['size'] for r in plan for f in r['files']),changed_samples=sum(bool(r['removed']) for r in results),backup_created=False)
    save_json(REPORT/'apply_report.json',dict(summary=summary,records=results));save_json(REPORT/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)

if __name__=='__main__':raise SystemExit(main())
