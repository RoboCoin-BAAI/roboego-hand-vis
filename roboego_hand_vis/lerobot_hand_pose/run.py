"""CLI shared by the L1 and L3 hand reconstruction adapters."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout, redirect_stderr
from threading import RLock, Event, Thread
import time
import os
from uuid import uuid4

from .compute import process, plans, existing_valid, run_logged
from .selection import from_pipeline_state, shallow_sessions
from roboego_hand_vis.lerobot_headband_integrity.L1.run import discover


def main(level, argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--model-root', required=True, type=Path, help='ego_hands_pose/video_reconstruction directory')
    p.add_argument('--model-python', type=Path, default=Path(sys.executable))
    devices = p.add_mutually_exclusive_group()
    devices.add_argument('--gpu', default='0')
    devices.add_argument('--gpus', help='Comma-separated GPU IDs; one session at a time per device')
    p.add_argument('--precision', choices=('auto','bf16','fp8'), default='auto')
    p.add_argument('--window-batch-size', type=int, default=1)
    p.add_argument('--resize-mode', choices=('long_edge_pad','center_crop'), default='long_edge_pad')
    p.add_argument('--ffmpeg-dir', type=Path)
    p.add_argument('--skip-existing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--sessions-file', type=Path)
    p.add_argument('--pipeline-state-root', type=Path, help='Legacy LEVEL_pipeline_state containing VIO logs')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--select-only', action='store_true', help='Save eligible paths without loading models')
    p.add_argument('--log-dir', type=Path, help='Unified standalone logs; default data/pipeline_tasks/LEVEL/hand_standalone')
    p.add_argument('--vio-success-only', action='store_true',
                   help='Only episodes whose current published VIO passed basic quality checks')
    args = p.parse_args(argv)
    if args.window_batch_size < 1:
        p.error('window-batch-size must be positive')
    log_dir = args.log_dir or args.source.resolve().parent / 'pipeline_tasks' / level / 'hand_standalone'
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / 'console.log'
    original = sys.stdout
    with path.open('a', encoding='utf-8', buffering=1) as file:
        tee = Tee(original, file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f'[hand-start] {time.strftime("%Y-%m-%d %H:%M:%S")} source={args.source}; unified log={path}', flush=True)
            stopped = Event()
            def heartbeat():
                while not stopped.wait(30):
                    print(f'[hand-alive] {time.strftime("%H:%M:%S")} job running; log={path}', flush=True)
            pulse = Thread(target=heartbeat, daemon=True)
            pulse.start()
            try:
                return execute(level, args, p)
            finally:
                stopped.set()
                pulse.join()


class Tee:
    def __init__(self, console, file):
        self.console, self.file, self.lock = console, file, RLock()
    def write(self, text):
        with self.lock:
            self.console.write(text)
            self.file.write(text)
            self.file.flush()
        return len(text)
    def flush(self):
        with self.lock:
            self.console.flush()
            self.file.flush()


def execute(level, args, p):
    print(f'[hand-scan] source={args.source}; recursive dataset scan disabled', flush=True)
    if args.sessions_file:
        sessions = discover(args.source, args.sessions_file)
    elif args.vio_success_only:
        sessions = from_pipeline_state(args.source, level, args.pipeline_state_root)
    else:
        sessions = shallow_sessions(args.source)
    if not sessions:
        p.error('No LeRobot sessions')
    print(f'[hand-scan] found {len(sessions)} sessions; VIO-only={args.vio_success_only}', flush=True)
    gpus = [v.strip() for v in args.gpus.split(',')] if args.gpus else [args.gpu]
    if any(not v for v in gpus) or len(set(gpus)) != len(gpus):
        p.error('GPU IDs must be nonempty and unique')
    pending, selection_errors, complete = [], 0, 0
    for index, session in enumerate(sessions, 1):
        print(f'[hand-select] {index}/{len(sessions)} {session}', flush=True)
        try:
            for plan in plans(session, level, vio_success_only=args.vio_success_only):
                if args.skip_existing and existing_valid(plan, args.model_root, args.precision,
                                                         args.window_batch_size, args.resize_mode):
                    complete += 1
                    continue
                pending.append({k:str(v) for k,v in plan.items()})
        except Exception as exc:
            selection_errors += 1
            print(json.dumps(dict(session=str(session),status='selection_failed',error=str(exc))),flush=True)
    log_dir=args.log_dir or args.source.resolve().parent / 'pipeline_tasks' / level / 'hand_standalone'
    request_root=log_dir / 'requests' / uuid4().hex
    request_root.mkdir(parents=True,exist_ok=True)
    selected=request_root/'selected.json'
    selected.write_text(json.dumps(pending,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[hand-selected] pending={len(pending)}, existing={complete}, selection_errors={selection_errors}; paths={selected}',flush=True)
    if args.dry_run or args.select_only or not pending:
        return int(bool(selection_errors))
    worker=Path(__file__).with_name('worker.py').resolve()
    def lane(index):
        assigned=pending[index::len(gpus)]
        if not assigned:
            return 0
        options=dict(model_root=str(args.model_root.resolve()),python=str(args.model_python),gpu=gpus[index],
                     precision=args.precision,window_batch_size=args.window_batch_size,resize_mode=args.resize_mode,
                     skip_existing=args.skip_existing,vio_success_only=args.vio_success_only)
        request=dict(level=level,tasks=assigned,options=options,results=str(request_root/f'gpu-{index}.results.jsonl'))
        path=request_root/f'gpu-{index}.json'
        path.write_text(json.dumps(request,indent=2),encoding='utf-8')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[index],PYTHONUNBUFFERED='1')
        if args.ffmpeg_dir:
            env['PATH']=str(args.ffmpeg_dir)+os.pathsep+env.get('PATH','')
        print(f'[hand-worker] GPU={gpus[index]}; tasks={len(assigned)}; one persistent model process',flush=True)
        result=run_logged([str(args.model_python),'-u',str(worker),'--request',str(path)],
                          cwd=args.model_root.resolve(),env=env,log_path=request_root/f'gpu-{index}.log',
                          label=f'hand-worker GPU={gpus[index]}')
        return int(bool(result.returncode))
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        failed = selection_errors + sum(pool.map(lane, range(len(gpus))))
    print(f'[hand-finished] episodes={len(pending)}, failed_workers_or_selections={failed}; records={request_root}',flush=True)
    return int(bool(failed))
