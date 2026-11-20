"""Generate upload README.md cards for the curated Jianzhi/Guanglun L1 collections."""
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

BASE=Path('/share/project/liuyou/humancoin/data')
REPO=Path('/share/project/liuyou/humancoin/code/roboego-hand-vis')
REPORT=BASE/'l1_dataset_readmes_20261119'
sys.path.insert(0,str(REPO))
from roboego_hand_vis.upload_data.readme_upload.gen_readme import (
    PORTAL_METADATA, PROJECT_CONFIG, generate_portal_yaml, read_json, render_l1_documents,
)
from roboego_hand_vis.upload_data.readme_upload._common import repo_id_for

def generate(job,project):
    path=Path(job['path']);result=dict(**job,status='done')
    try:
        text,metadata=render_l1_documents(path,project)
        if f'| Tasks | {job["tasks"]} |' not in text:raise ValueError('Task count mismatch')
        if f'# {path.name}\n' not in text or f'| Repository ID | HumanCOIN/{path.name} |' not in text:raise ValueError('Repository name mismatch')
        if '| Episodes |' in text or 'Samples (metadata)' in text:raise ValueError('Single-recording statistics in collection card')
        if 'viewer: false' not in text or 'root=task_dir' not in text:raise ValueError('Invalid L1 loading/viewer configuration')
        documents = {'README.md': text, PORTAL_METADATA: generate_portal_yaml(path, metadata)}
        for name, content in documents.items():
            payload = content.encode('utf-8')
            output = path / name
            fd, tmp = tempfile.mkstemp(prefix=f'.{name}.', dir=path)
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, output)
            finally:
                Path(tmp).unlink(missing_ok=True)
            if output.read_bytes() != payload:
                raise ValueError(f'Readback failed: {output}')
        readme = text.encode('utf-8')
        result.update(bytes=len(readme),sha256=hashlib.sha256(readme).hexdigest())
    except Exception as exc:result.update(status='error',error=str(exc))
    return result

def main():
    roots={'jianzhi':BASE/'jianzhi/wait_for_publish/publish','guanglun':BASE/'guanglun_depth/publish'}
    jz=json.loads((BASE/'jianzhi_publish_dataset_move_20261119/plan.json').read_text())['records']
    gl=json.loads((BASE/'guanglun_badfps_move_01_49_20261119/selection.json').read_text())['retained']
    counts={'jianzhi':Counter(r['dataset_name'] for r in jz),'guanglun':Counter(r['dataset_name'] for r in gl)}
    if len(jz)!=7078 or len(gl)!=4338:raise ValueError('Unexpected task scope')
    jobs=[]
    for dataset,root in roots.items():
        actual={p.name for p in root.iterdir() if p.is_dir()}
        if actual!=set(counts[dataset]):raise ValueError('Collection directory scope mismatch: '+dataset)
        jobs.extend(dict(dataset=dataset,path=str(root/name),tasks=n) for name,n in sorted(counts[dataset].items()))
    if len(jobs)!=463 or len({repo_id_for(Path(r['path']),'HumanCOIN') for r in jobs})!=463:raise ValueError('Missing/duplicate repository names')
    REPORT.mkdir(exist_ok=True);project=read_json(PROJECT_CONFIG);results=[];start=time.monotonic()
    with (REPORT/'progress.jsonl').open('w') as log,ThreadPoolExecutor(8) as pool:
        futures=[pool.submit(generate,job,project) for job in jobs]
        for future in as_completed(futures):
            r=future.result();results.append(r);log.write(json.dumps(r)+'\n');log.flush()
            if len(results)%25==0 or len(results)==len(jobs):print(json.dumps(dict(completed=len(results),total=len(jobs),tasks=sum(r['tasks'] for r in results),errors=sum(r['status']=='error' for r in results),elapsed_s=round(time.monotonic()-start))),flush=True)
    summary=dict(cards=len(results),tasks=sum(r['tasks'] for r in results),errors=sum(r['status']=='error' for r in results),seconds=round(time.monotonic()-start,2),datasets={ds:dict(cards=sum(r['dataset']==ds for r in results),tasks=sum(r['tasks'] for r in results if r['dataset']==ds),errors=sum(r['dataset']==ds and r['status']=='error' for r in results)) for ds in roots},uploaded=False)
    (REPORT/'results.json').write_text(json.dumps({'summary':summary,'records':results},indent=2));(REPORT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)

if __name__=='__main__':raise SystemExit(main())
