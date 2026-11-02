"""Run depth then hand on rectified sessions absent from batch done lists."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Lock
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def load(path):
    values = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(values,list) or any(not isinstance(x,str) for x in values):
        raise ValueError('Expected path array: '+str(path))
    return set(values)


def save(path, values):
    temp=path.with_name('.'+path.name+'.'+uuid4().hex)
    temp.write_text(json.dumps(sorted(values),indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)


class Worker:
    def __init__(self, stage, device, log_dir):
        self.stderr = (log_dir/(stage+'-gpu-'+device+'-'+uuid4().hex+'.log')).open('a')
        self.process = subprocess.Popen([sys.executable,'-u',
            str(Path(__file__).with_name('gpu_supplement_worker.py')),stage,'L1',device],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            text=True,encoding='utf-8',bufsize=1)

    def execute(self, command, log):
        self.process.stdin.write(json.dumps(dict(command=command,log=str(log)))+'\n')
        self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:
            raise RuntimeError('Resident worker exited; see GPU log')
        return json.loads(line)['returncode']

    def close(self):
        try:
            self.process.stdin.close()
        except (OSError, BrokenPipeError):
            pass
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()
        self.process.stdout.close()
        self.stderr.close()


def run_phase(stage, tasks, devices, log_dir, command, worker_factory=Worker):
    """One resident per lane, draining a single queue spanning all batches."""
    queue=Queue()
    for session,done_file in tasks:
        queue.put((session,done_file))
    guard=Lock()
    def lane(device):
        worker=None
        failures=0
        try:
            while True:
                try:
                    session,done_file=queue.get_nowait()
                except Empty:
                    break
                log=log_dir/stage/Path(session).parent.name/(Path(session).name+'.log')
                log.parent.mkdir(parents=True,exist_ok=True)
                try:
                    if worker is None:
                        worker=worker_factory(stage,device,log_dir)
                    print(f'{stage} GPU {device}: {session}',flush=True)
                    rc=worker.execute(command(stage,session,device),log)
                    if rc!=0:
                        raise RuntimeError(f'Exit {rc}; log: {log}')
                    with guard:
                        save(done_file,load(done_file)|{session})
                except Exception as exc:
                    failures+=1
                    print(f'FAILED {stage} {session}: {exc}',flush=True)
                    # Keep a healthy resident alive after an individual session failure.
                    if worker is not None and getattr(worker,'process',None) is not None and worker.process.poll() is not None:
                        worker.close()
                        worker=None
        finally:
            if worker is not None:
                worker.close()
        return failures
    with ThreadPoolExecutor(max_workers=min(len(devices),len(tasks))) as pool:
        return sum(pool.map(lane, devices[:len(tasks)]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/share/project/liuyou/humancoin/data/L1_lerobot'))
    p.add_argument('--batches',nargs='+')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--depth-workers-per-gpu',type=int,default=2)
    p.add_argument('--stages',nargs='+',choices=('depth','hand'),default=['depth','hand'])
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    devices=a.gpus.split(',')
    if any(not d.isdigit() for d in devices) or len(set(devices))!=len(devices) or a.depth_workers_per_gpu<1:
        p.error('GPU IDs must be unique integers; depth workers must be positive')
    repo=Path(__file__).resolve().parents[2]
    root=a.root.resolve()
    indices=([root/b/'rectification_done.json' for b in a.batches] if a.batches
             else sorted(root.glob('batch_*/rectification_done.json')))
    if not indices: p.error('No batch rectification_done.json files yet')
    work=[]
    for index in indices:
        if index.parent.parent!=root: p.error('Invalid batch')
        ready=load(index)
        for s in ready:
            if Path(s).resolve().parent!=index.parent.resolve(): p.error('Session outside batch: '+s)
        for stage in ('depth','hand'):
            done_file=index.parent/(stage+'_done.json')
            done=load(done_file)  # Require the existing inventory, never silently ignore it.
            work.append((stage,index.parent,done_file,ready-done))
    def command(stage, session, device):
        if stage=='depth':
            return [sys.executable,'depth','--source',session,
                '--model-root',str(repo.parent/'Fast-FoundationStereo'),
                '--ckpt',str(repo.parent/'Fast-FoundationStereo/weights/model_best_bp2_serialize.pth'),
                '--engine','throughput','--num-frames','auto','--depth-format','hevc12',
                '--ffmpeg',str(Path(sys.executable).parent/'ffmpeg'),'--no-skip-existing']
        return [sys.executable,'hand','--source',session,
            '--model-root',str(repo.parent/'ego_hands_pose/video_reconstruction'),
            '--window-batch-size','1','--ffmpeg-dir',str(Path(sys.executable).parent),
            '--no-skip-existing']
    merged={stage:sorted({(s,done_file) for name,batch,done_file,pending in work
                         if name==stage for s in pending}) for stage in ('depth','hand')}
    for stage,tasks in merged.items():
        print(f'{stage}: {len(tasks)} sessions across all selected batches',flush=True)
    if not a.execute:
        return 0
    if 'depth' in a.stages and merged['depth']:
        checkpoint=repo.parent/'Fast-FoundationStereo/weights/model_best_bp2_serialize.pth'
        for required in (checkpoint,checkpoint.with_name('cfg.yaml')):
            if not required.is_file():
                p.error('Missing depth model file: '+str(required))
    log_dir=root.parent/'pipeline_tasks/L1/gpu_supplement'/uuid4().hex
    log_dir.mkdir(parents=True)
    print('Logs: '+str(log_dir),flush=True)
    for stage,tasks in merged.items():
        save(log_dir/(stage+'_pending.json'),{s for s,_ in tasks})
    failed=0
    for stage in ('depth','hand'):
        if stage not in a.stages:
            continue
        tasks=merged[stage]
        if not tasks:
            continue
        slots=[device for device in devices for _ in range(a.depth_workers_per_gpu if stage=='depth' else 1)]
        failed+=run_phase(stage,tasks,slots,log_dir,command)
        # run_phase waits for every worker to exit before the other model is loaded.
    return int(bool(failed))



if __name__=='__main__':
    raise SystemExit(main())
