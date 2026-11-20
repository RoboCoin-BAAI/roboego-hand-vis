"""Align only video timestamp tables and master index, with narrow meta updates."""
import copy
import io
import json
import stat
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import av
import pyarrow as pa
import pyarrow.parquet as pq
from apply_scene_classification import atomic_write
from reclassify_scenes_qwen import save_json
from audit_jianzhi_video_rows import BASE, ROOT, KEYS, REPORT as AUDIT

REPORT=BASE/'jianzhi_video_rows_repair_20261119'

def resize(table, count, fps):
    if count<1 or len(table)<1:raise ValueError('Empty timeline')
    if count<=len(table):return table.slice(0,count)
    last=table.slice(len(table)-1).to_pylist()[0]
    rows=[]
    for offset in range(1,count-len(table)+1):
        row=dict(last)
        for key in row:
            if key=='timestamp':row[key]=last[key]+offset/fps
            elif key in ('frame_index','index'):row[key]=last[key]+offset
            elif key not in ('episode_index','task_index'):raise ValueError('Unsupported padding column: '+key)
        rows.append(row)
    return pa.concat_tables([table,pa.Table.from_pylist(rows,schema=table.schema)])

def json_bytes(data):return (json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')

def repair(audit):
    session=ROOT/audit['path']
    result=dict(path=audit['path'],status='done',files=[],streams={})
    try:
        if session.is_symlink() or ROOT.resolve() not in session.resolve().parents:raise ValueError('Unsafe session')
        staged={};targets={};tables={}
        def stage(path,payload):
            if path.is_symlink():raise ValueError('Symlink target')
            raw=path.read_bytes()
            if raw!=payload:staged[path]=(raw,payload)
        def table_stage(path,target,fps):
            old=pq.read_table(path);new=resize(old,target,fps)
            if len(new)!=len(old):
                if not old.slice(0,min(len(old),target)).equals(new.slice(0,min(len(old),target))):raise ValueError('Prefix changed')
                sink=pa.BufferOutputStream();pq.write_table(new,sink,compression='snappy')
                payload=sink.getvalue().to_pybytes()
                if not pq.read_table(pa.BufferReader(payload)).equals(new,check_metadata=True):raise ValueError('Parquet roundtrip failed')
                stage(path,payload)
            return old,new
        for key in KEYS:
            stream=audit['streams'][key]
            if len(stream['videos'])!=1 or len(stream['parquets'])!=1:raise ValueError('Expected single episode per stream')
            video=stream['videos'][0];target=video['frames'];fps=video['fps']
            with av.open(str(session/video['path'])) as c:
                if c.streams.video[0].frames!=target:raise ValueError('Video changed since audit')
            p=session/stream['parquets'][0]['path']
            old,new=table_stage(p,target,fps)
            tables[key]=(old,new)
            targets[key]=target
            result['streams'][key]=dict(before=len(old),after=target,delta=target-len(old))
        left=targets[KEYS[0]];fps=audit['streams'][KEYS[0]]['videos'][0]['fps']
        if len(audit['master'])!=1:raise ValueError('Expected one master index')
        master_path=session/audit['master'][0]['path']
        old_master,new_master=table_stage(master_path,left,fps)
        tables['master_index']=(old_master,new_master)
        if len(new_master)!=left:raise ValueError('Master length mismatch')
        ip=session/'meta/info.json';info=json.loads(ip.read_bytes().decode('utf-8-sig'))
        updated=copy.deepcopy(info);updated['total_frames']=left
        if info!=updated:stage(ip,json_bytes(updated))
        ep=session/'meta/episodes.jsonl'
        episodes=[json.loads(line) for line in ep.read_text().splitlines() if line.strip()]
        if len(episodes)!=1 or episodes[0]['episode_index']!=0:raise ValueError('Expected single episode 0')
        revised=copy.deepcopy(episodes);revised[0]['length']=left
        if episodes!=revised:stage(ep,(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in revised)).encode())
        # These files describe independent sensor streams, not the master timeline.
        for name in ('episodes_stats.jsonl','stats.json'):
            path=session/'meta'/name
            if not path.exists():continue
            stats=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if name.endswith('jsonl') else [json.loads(path.read_text())]
            for row in stats:
                if set(row.get('stats',row)) & set(KEYS+['timestamp','frame_index','index','master_index']):
                    raise ValueError('Affected statistics need explicit recomputation: '+name)
        rp=session/'meta/readme.json'
        if rp.exists():
            readme=json.loads(rp.read_bytes().decode('utf-8-sig'));revised=copy.deepcopy(readme)
            revised['frame_count']=left
            revised['duration_s']=left/info['fps']
            for c in revised.get('cameras',{}).get('items',[]):
                if c.get('id') in targets:c['sample_count']=targets[c['id']]
            files=[p for p in session.rglob('*') if p.is_file() and p!=rp and not (p.parent==rp.parent and p.name.startswith('.readme.json.'))]
            revised['file_count']=len(files)
            revised['size_bytes']=sum(len(staged[p][1]) if p in staged else p.stat().st_size for p in files)
            if readme!=revised:stage(rp,json_bytes(revised))
        # Prepare and validate everything before replacing any source file.
        for path,(raw,payload) in staged.items():
            if path.read_bytes()!=raw:raise ValueError('Concurrent change: '+str(path))
        for path,(raw,payload) in staged.items():
            atomic_write(path,payload,stat.S_IMODE(path.stat().st_mode))
            if path.read_bytes()!=payload:raise ValueError('Write verification failed')
            result['files'].append(dict(path=str(path.relative_to(session)),byte_delta=len(payload)-len(raw)))
        for key,(_,expected) in tables.items():
            path=master_path if key=='master_index' else session/audit['streams'][key]['parquets'][0]['path']
            if not pq.read_table(path).equals(expected,check_metadata=True):raise ValueError('Final table verification failed')
        result['master_before']=len(old_master);result['master_after']=left
    except Exception as exc:result.update(status='error',error=str(exc))
    return result

def main():
    audit=json.loads((AUDIT/'audit.json').read_text())
    if audit['summary']['errors']:raise ValueError('Audit has errors')
    rows=[r for r in audit['records'] if r['mismatch'] or r['info_frames']!=sum(v['frames'] for v in r['streams'][KEYS[0]]['videos']) or sum(p['rows'] for p in r['master'])!=sum(v['frames'] for v in r['streams'][KEYS[0]]['videos'])]
    REPORT.mkdir(parents=True,exist_ok=True);results=[]
    with (REPORT/'progress.jsonl').open('w') as log,ThreadPoolExecutor(8) as pool:
        for future in as_completed([pool.submit(repair,row) for row in rows]):
            r=future.result();results.append(r);log.write(json.dumps(r)+'\n');log.flush()
            if len(results)%50==0:print(json.dumps(dict(completed=len(results),total=len(rows),errors=sum(r['status']=='error' for r in results))),flush=True)
    summary=dict(audited=audit['summary']['samples'],selected=len(rows),updated=sum(bool(r['files']) for r in results),errors=sum(r['status']=='error' for r in results),backups=False)
    save_json(REPORT/'apply_report.json',dict(summary=summary,records=results));print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)

if __name__=='__main__':raise SystemExit(main())
