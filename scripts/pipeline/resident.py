"""JSON-lines resident GPU worker; stdout is reserved for completion messages."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def options(argv):
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--ckpt',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--gpus',default='0')
    p.add_argument('--gpu',default='0')
    p.add_argument('--model-python')
    p.add_argument('--window-batch-size',type=int,default=1)
    p.add_argument('--ffmpeg-dir',type=Path)
    p.add_argument('--ffmpeg',default='ffmpeg')
    p.add_argument('--num-frames',default='auto')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--engine',default='throughput')
    p.add_argument('--depth-format',default='hevc12')
    p.add_argument('--no-skip-existing',action='store_true')
    return p.parse_args(argv[2:])


def main():
    stage,level,device=sys.argv[1:]
    os.environ['CUDA_VISIBLE_DEVICES']=device
    os.environ['OMP_NUM_THREADS']='2'
    os.environ['MKL_NUM_THREADS']='2'
    model=None
    holder=[None]
    for line in sys.stdin:
        request=json.loads(line)
        code=0
        with open(request['log'],'a',encoding='utf-8',buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                a=options(request['command'])
                started=time.perf_counter()
                if stage=='depth':
                    from roboego_hand_vis.lerobot_fast_foundation_depth.L1.engine import DepthEngine
                    from roboego_hand_vis.lerobot_fast_foundation_depth.L1.compute import DEFAULT_CONFIG,plan_session
                    cameras=None
                    if level=='L3':
                        from roboego_hand_vis.lerobot_fast_foundation_depth.L3.compute import DEFAULT_CONFIG,CAMERAS
                        cameras=CAMERAS
                    from roboego_hand_vis.lerobot_fast_foundation_depth.L1.reuse import existing_valid
                    for plan in plan_session(a.source,camera_names=cameras):
                        if not a.no_skip_existing and existing_valid(plan):
                            print(json.dumps(dict(episode=plan['episode'],status='done',reused=True)),flush=True)
                            continue
                        if model is None:
                            model=DepthEngine(dict(config=str(a.config or DEFAULT_CONFIG),model_root=str(a.model_root),
                                ckpt=str(a.ckpt) if a.ckpt else None,start=0,num_frames=a.num_frames,
                                camera_names=cameras,cpu_threads=2,prefetch=4,write_queue=4,
                                batch_size=a.batch_size,
                                depth_format=a.depth_format,ffmpeg=a.ffmpeg,save_vis=False))
                            print(f'[resident-load] gpu={device} model_load_s={time.perf_counter()-started:.3f}',flush=True)
                        print(json.dumps(model.process(dict(session=str(a.source),episode=plan['episode']))),flush=True)
                else:
                    if a.ffmpeg_dir:
                        os.environ['PATH']=str(a.ffmpeg_dir)+os.pathsep+os.environ.get('PATH','')
                    from roboego_hand_vis.lerobot_hand_pose.compute import plans,process
                    def infer(video,output,log_path):
                        # Loaded lazily, only when process determines an episode is not reusable.
                        if holder[0] is None:
                            sys.path.insert(0,str(a.model_root.resolve()))
                            os.chdir(a.model_root)
                            from runtime.pipeline import Predictor
                            holder[0]=Predictor(device='cuda:0',precision='auto')
                            holder[0].config['window_batch_size']=a.window_batch_size
                            print(f'[resident-load] gpu={device} model_load_s={time.perf_counter()-started:.3f}',flush=True)
                        with log_path.open('w',encoding='utf-8') as task_log, redirect_stdout(task_log),redirect_stderr(task_log):
                            holder[0].run(video,output,resize_mode='long_edge_pad',max_frames=0,render_overlay=False)
                        return SimpleNamespace(returncode=0)
                    result=process(a.source,level,model_root=a.model_root,python=sys.executable,gpu=device,
                        precision='auto',window_batch_size=a.window_batch_size,resize_mode='long_edge_pad',
                        skip_existing=not a.no_skip_existing,vio_success_only=False,
                        selected_plans=plans(a.source,level),inference=infer)
                    print(json.dumps(result,default=str),flush=True)
            except Exception:
                traceback.print_exc()
                code=1
        print(json.dumps(dict(returncode=code)),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
