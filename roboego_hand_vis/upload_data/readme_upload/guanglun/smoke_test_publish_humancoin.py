"""Read-only HumanCOIN/PyAV checks of five Jianzhi and five Guanglun samples."""
import os
from pathlib import Path
BASE=Path('/share/project/liuyou/humancoin/data')
REPORT=BASE/'humancoin_loadcheck_20261119'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['HF_DATASETS_CACHE']=str(REPORT/'datasets_cache')
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib
import importlib.metadata
import json
import random
import sys
import time
import traceback

sys.path.insert(0,'/share/project/liuyou/humancoin/code/roboego-hand-vis')
from roboego_hand_vis.upload_data.readme_upload.check_load import validate_dataset_load

def select():
    rng=random.Random(20261119)
    jz=json.loads((BASE/'jianzhi_publish_dataset_move_20261119/plan.json').read_text())['records']
    old=json.loads((BASE/'jianzhi_video_rows_audit_20261119/audit.json').read_text())['records']
    shorter={Path(r['path']).name for r in []}
    shorter={Path(r['path']).name for r in old if r['status']=='done' and r['streams']['observation.images.head_right_rectified']['videos'][0]['frames']<r['streams']['observation.images.head_left_rectified']['videos'][0]['frames']}
    special=[r for r in jz if Path(r['original_source']).name in shorter]
    chosen=[rng.choice(special)] if special else []
    chosen+=rng.sample([r for r in jz if r not in chosen],5-len(chosen))
    samples=[dict(dataset='jianzhi',path=r['target']) for r in chosen]
    gl=json.loads((BASE/'guanglun_badfps_move_01_49_20261119/selection.json').read_text())['retained']
    for fps,n in [(30.0,3),(25.0,2)]:
        samples.extend(dict(dataset='guanglun',path=r['source']) for r in rng.sample([r for r in gl if float(r['fps'])==fps],n))
    return samples

def main():
    REPORT.mkdir(exist_ok=True);samples=select()
    (REPORT/'sample_selection.json').write_text(json.dumps(samples,indent=2))
    print('Importing HumanCOIN; ten sample roots selected',flush=True)
    from humancoin import HumanCOINDataset
    import torch
    torch.set_num_threads(1)
    results=[]
    for i,row in enumerate(samples,1):
        root=Path(row['path']);start=time.monotonic();event=dict(**row,status='done',frames=[])
        monitored=[p for p in (root/'meta').iterdir() if p.is_file()]+list((root/'annotation').glob('episode_*_structured.json'))
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in monitored}
        info=json.loads((root/'meta/info.json').read_text());readme=json.loads((root/'meta/readme.json').read_text())
        class RecordingDataset:
            def __init__(self,**kwargs):self.dataset=HumanCOINDataset(**kwargs)
            def __len__(self):return len(self.dataset)
            def __getitem__(self,index):
                sample=self.dataset[index]
                keys={k for k in info['features'] if k not in ('timestamp','episode_index','frame_index','index','task_index')}
                if keys-set(sample):raise ValueError('Missing streams: '+str(sorted(keys-set(sample))))
                detail=dict(index=index,streams={})
                for key in sorted(keys):
                    value=sample[key]
                    d={'shape':list(value.shape),'dtype':str(value.dtype)} if hasattr(value,'shape') else {'type':type(value).__name__}
                    feature=info['features'][key]
                    if feature.get('dtype')=='video':
                        h,w,c=feature['shape']
                        if list(value.shape)!=[c,h,w]:raise ValueError('Video shape mismatch: '+key+' '+str(value.shape))
                        if 'depth' in key:
                            if value.dtype not in (torch.uint16,torch.int16,torch.int32,torch.int64):raise ValueError('Depth was converted to RGB/normalized float')
                            numeric=value.to(torch.int32);d.update(min=int(numeric.min()),max=int(numeric.max()))
                    detail['streams'][key]=d
                event['frames'].append(detail)
                return sample
        try:
            validate_dataset_load(root,'HumanCOIN/'+readme['dataset_name'],info,loader=RecordingDataset)
        except Exception as exc:event.update(status='error',error=str(exc),traceback=traceback.format_exc())
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v for p,v in hashes.items()):event.update(status='error',error='Source metadata changed during load')
        event['seconds']=round(time.monotonic()-start,2);results.append(event)
        (REPORT/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(dict(completed=i,total=len(samples),dataset=row['dataset'],path=row['path'],status=event['status'],error=event.get('error'),seconds=event['seconds'])),flush=True)
    summary=dict(samples=len(results),passed=sum(r['status']=='done' for r in results),errors=sum(r['status']=='error' for r in results),humancoin_version=importlib.metadata.version('humancoin'),datasets={ds:dict(samples=sum(r['dataset']==ds for r in results),passed=sum(r['dataset']==ds and r['status']=='done' for r in results)) for ds in ['jianzhi','guanglun']})
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)

if __name__=='__main__':raise SystemExit(main())
