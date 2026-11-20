"""Read-only audit of published Jianzhi video sample tables."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import av
import pyarrow.parquet as pq
from reclassify_scenes_qwen import save_json

BASE = Path('/share/project/liuyou/humancoin/data')
ROOT = BASE/'jianzhi/wait_for_publish'
REPORT = BASE/'jianzhi_video_rows_audit_20261119'
KEYS = ['observation.images.head_left_rectified', 'observation.images.head_right_rectified', 'observation.images.depth_head_left_rectified']

def audit(row):
    session = (ROOT/row['path']).parent.parent
    result = dict(path=str(session.relative_to(ROOT)), status='done', streams={})
    try:
        info=json.loads((session/'meta/info.json').read_text())
        result['info_frames']=info.get('total_frames')
        result['episodes']=[json.loads(line) for line in (session/'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        for key in KEYS:
            videos=sorted((session/'videos').glob('chunk-*/'+key+'/episode_*.mp4'))
            tables=sorted((session/'data').glob('chunk-*/episode_*/'+key+'.parquet'))
            frames=[]
            for video in videos:
                with av.open(str(video)) as container:
                    stream=container.streams.video[0]
                    if stream.frames<=0:raise ValueError('No container frame count: '+str(video))
                    frames.append(dict(path=str(video.relative_to(session)),frames=stream.frames,fps=float(stream.average_rate)))
            if not videos or not tables:raise ValueError('Missing stream '+key)
            result['streams'][key]=dict(videos=frames,parquets=[dict(path=str(p.relative_to(session)),rows=pq.ParquetFile(p).metadata.num_rows) for p in tables])
        result['master']=[dict(path=str(p.relative_to(session)),rows=pq.ParquetFile(p).metadata.num_rows) for p in sorted((session/'data').glob('chunk-*/episode_*/master_index.parquet'))]
        result['mismatch']=any(sum(v['frames'] for v in stream['videos'])!=sum(p['rows'] for p in stream['parquets']) for stream in result['streams'].values())
    except Exception as exc:result.update(status='error',error=str(exc))
    return result

def main():
    rows=[r for r in json.loads((BASE/'common_verb_lt5_apply_20261119/apply_report.json').read_text())['records'] if r['dataset']=='jianzhi']
    assert len(rows)==7078
    REPORT.mkdir(parents=True,exist_ok=True)
    results=[]
    with (REPORT/'progress.jsonl').open('w') as log,ThreadPoolExecutor(12) as pool:
        for f in as_completed([pool.submit(audit,row) for row in rows]):
            result=f.result();results.append(result)
            log.write(json.dumps(result)+'\n');log.flush()
            if len(results)%500==0:print(json.dumps(dict(completed=len(results),total=len(rows),errors=sum(r['status']=='error' for r in results),mismatches=sum(r.get('mismatch',False) for r in results))),flush=True)
    deltas={key:Counter() for key in KEYS}
    for r in results:
        if r['status']!='done':continue
        for key,stream in r['streams'].items():
            deltas[key][sum(v['frames'] for v in stream['videos'])-sum(p['rows'] for p in stream['parquets'])]+=1
    summary=dict(samples=len(results),errors=sum(r['status']=='error' for r in results),mismatches=sum(r.get('mismatch',False) for r in results),video_minus_parquet={k:dict(v) for k,v in deltas.items()})
    save_json(REPORT/'audit.json',dict(summary=summary,records=results))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
