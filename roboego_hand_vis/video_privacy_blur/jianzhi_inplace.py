"""Four-GPU, resumable in-place face blur for Jianzhi published sessions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

from .batch import directory_lock, sha256, write_json

CAMERAS = ('observation.images.head_left_rectified', 'observation.images.head_right_rectified')


def discover(root, start, end):
    videos, errors = [], []
    for number in range(start, end + 1):
        batch = root / f'batch_{number:07d}'
        if not batch.is_dir() or batch.is_symlink():
            errors.append(f'Missing or symlink batch: {batch}')
            continue
        sessions = sorted(p for p in batch.iterdir() if p.is_dir() and not p.is_symlink() and len(p.name) == 32 and all(c in '0123456789abcdef' for c in p.name))
        if not sessions:
            errors.append(f'No sessions in {batch}')
        for session in sessions:
            for camera in CAMERAS:
                video = session / 'videos' / 'chunk-000' / camera / 'episode_000000.mp4'
                if not video.is_file() or any(p.is_symlink() for p in (video, *video.parents) if p != root and root in p.parents):
                    errors.append(f'Missing or symlink video: {video}')
                else:
                    videos.append(video)
    return videos, errors


class ResidentWorker:
    def __init__(self, gpu, state, config, lock_fd):
        self.gpu, self.state, self.config = gpu, state, config
        self.lock_fd = lock_fd
        self.process = None
        self.log = None

    def run(self, request):
        if self.process is None or self.process.poll() is not None:
            self.close()
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=self.gpu, OMP_NUM_THREADS=str(self.config['threads']),
                       MKL_NUM_THREADS=str(self.config['threads']), OPENBLAS_NUM_THREADS=str(self.config['threads']))
            self.log = (self.state / 'logs' / ('gpu-' + self.gpu.replace('/', '_') + '.log')).open('a', encoding='utf-8')
            self.process = subprocess.Popen([sys.executable, '-m', 'roboego_hand_vis.video_privacy_blur.worker', '--persistent'],
                                            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, text=True, bufsize=1, pass_fds=(self.lock_fd,))
        self.process.stdin.write(json.dumps({'request': str(request)}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Resident worker exited; see {self.log.name}')
        response = json.loads(line)
        if not response.get('ok'):
            raise RuntimeError(f"{response.get('error')}; see {self.log.name}")

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.stdin.close()
                self.process.wait()
            self.process.stdout.close()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


def process_video(source, state, config, gpu, worker):
    key = hashlib.sha256(str(source).encode()).hexdigest()
    record_path = state / 'jobs' / (key + '.json')
    log = state / 'logs' / (key + '.log')
    temporary = source.with_name('.' + source.stem + '.privacy-' + key[:12] + '.mp4')
    request = state / 'work' / (key + '.request.json')
    report = temporary.with_suffix('.report.json')
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    current = sha256(source)
    if record:
        if record.get('source') != str(source):
            raise RuntimeError('Job record source mismatch')
        if record.get('status') in ('prepared', 'complete') and current == record.get('output_sha256'):
            record['status'] = 'complete'
            write_json(record_path, record)
            return 'skipped'
        if record.get('status') == 'complete' or (record.get('input_sha256') and current != record['input_sha256']):
            raise RuntimeError('Source changed since previous run; inspect job record')
        if record.get('signature') != signature:
            raise RuntimeError('Parameters changed since previous run; inspect job record')
    record = {'source': str(source), 'input_sha256': current, 'signature': signature, 'status': 'running', 'gpu': gpu}
    write_json(record_path, record)
    temporary.unlink(missing_ok=True)
    report.unlink(missing_ok=True)
    write_json(request, {'source': str(source), 'output': str(temporary), 'config': config})
    with log.open('a', encoding='utf-8') as stream:
        stream.write(f'\nStarting on GPU {gpu}\n')
    worker.run(request)
    details = json.loads(report.read_text())
    if details.get('status') != 'complete' or details.get('frames', 0) <= 0 or not temporary.is_file() or not temporary.stat().st_size:
        raise RuntimeError('Invalid worker output')
    if sha256(source) != current:
        raise RuntimeError('Source changed during processing')
    record.update(status='prepared', output_sha256=sha256(temporary), details=details)
    write_json(record_path, record)
    os.replace(temporary, source)
    record['status'] = 'complete'
    write_json(record_path, record)
    report.unlink(missing_ok=True)
    return 'completed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/share/project/liuyou/humancoin/data/jianzhi/wait_for_publish'))
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=71)
    parser.add_argument('--gpus', default='0,1,2,3', help='CUDA-visible device indices')
    parser.add_argument('--weights', type=Path, default=Path(__file__).resolve().parents[2] / 'conf/models/yolov8n-face.pt')
    parser.add_argument('--state-dir', type=Path, required=True, help='Persistent run state outside the input tree')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    state = args.state_dir.resolve()
    gpus = args.gpus.split(',')
    if args.start < 1 or args.end < args.start or args.threads < 1 or any(not x.isdigit() for x in gpus) or len(set(gpus)) != len(gpus):
        parser.error('Invalid range, GPU list, or thread count')
    if state == root or root in state.parents:
        parser.error('State directory must be outside the input tree')
    videos, errors = discover(root, args.start, args.end)
    print(json.dumps({'videos': len(videos), 'discovery_errors': errors}, ensure_ascii=False), flush=True)
    if errors or not videos:
        return 1
    if args.dry_run:
        return 0
    import torch
    if not torch.cuda.is_available() or max(map(int, gpus)) >= torch.cuda.device_count():
        parser.error(f'Requested GPUs {gpus}; PyTorch sees {torch.cuda.device_count()} CUDA devices')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible is not None else None
    worker_gpus = [devices[int(gpu)].strip() if devices is not None else gpu for gpu in gpus]
    weights = args.weights.resolve(strict=True)
    config = dict(face_weights=str(weights), weights_sha256=sha256(weights), device='cuda', threads=args.threads,
                  conf=.35, imgsz=960, scale=1.35, blur_strength=1., method='gaussian', track_gap=0, track_scale=1.25)
    for name in ('jobs', 'logs', 'work'):
        (state / name).mkdir(parents=True, exist_ok=True)
    jobs = queue.Queue()
    for video in videos:
        jobs.put(video)
    counts = {'completed': 0, 'skipped': 0, 'failed': 0}
    def run_gpu(gpu, lock_fd):
        local = {'completed': 0, 'skipped': 0, 'failed': 0}
        worker = ResidentWorker(gpu, state, config, lock_fd)
        try:
            while True:
                try:
                    video = jobs.get_nowait()
                except queue.Empty:
                    return local
                try:
                    status = process_video(video, state, config, gpu, worker)
                    local[status] += 1
                    print(f'[{status}] GPU {gpu}: {video}', flush=True)
                except Exception as exc:
                    local['failed'] += 1
                    print(f'[failed] GPU {gpu}: {video}: {exc}', file=sys.stderr, flush=True)
        finally:
            worker.close()
    with directory_lock(state / 'lock') as lock_fd:
        with ThreadPoolExecutor(max_workers=len(worker_gpus)) as executor:
            for partial in executor.map(lambda gpu: run_gpu(gpu, lock_fd), worker_gpus):
                for name in counts:
                    counts[name] += partial[name]
        write_json(state / 'summary.json', counts)
    print(json.dumps(counts), flush=True)
    return int(counts['failed'] > 0)


if __name__ == '__main__':
    sys.exit(main())
