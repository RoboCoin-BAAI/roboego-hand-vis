"""Estimate Jianzhi depth locally or watch pipeline upload receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
from queue import Queue, Empty
import shutil
import sys
from threading import Event, Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.L1.gpu_supplement import Worker
from roboego_hand_vis.upload_data.ks3 import KS3, atomic_json, prefix


def load(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def discover(root, first, last):
    """Jianzhi records remain ready after the conversion pipeline cleans local data."""
    for records in sorted(root.glob('batch_*/records')):
        batch = records.parent.name
        if not batch[6:].isdigit() or not first <= int(batch[6:]) <= last:
            continue
        done = load(records.parent / 'depth_done.json', [])
        if not isinstance(done, list) or any(not isinstance(uri, str) for uri in done):
            raise ValueError('Expected URI array in ' + str(records.parent))
        for path in sorted(records.glob('*.json')):
            row = load(path, {})
            if row.get('status') not in ('uploaded', 'complete') or row.get('error'):
                continue
            uri = row.get('destination')
            if not isinstance(uri, str) or not uri.startswith('ks3://'):
                raise ValueError('Missing KS3 destination in ' + str(path))
            if uri not in done:
                yield batch, dict(row, destination_uri=uri)


def download_dataset(client, row, target):
    destination = prefix(row['destination_uri'])
    sizes = row['upload']['file_sizes']
    if not sizes or 'meta/info.json' not in sizes:
        raise ValueError('Missing upload file receipt')
    from scripts.jianzhi.download_depth import select_objects
    selected = select_objects([dict(uri=destination+name, size=size, etag='')
                               for name, size in sizes.items()], destination)
    for item in selected:
        name, size = item['relative'], item['size']
        parts = PurePosixPath(name).parts
        if not parts or name.startswith('/') or '\\' in name or any(p in ('.', '..') for p in parts):
            raise ValueError('Unsafe relative object key: '+name)
        path = target.joinpath(*parts)
        if not path.resolve().is_relative_to(target.resolve()):
            raise ValueError('Object escapes download directory')
        if path.is_file() and path.stat().st_size == size:
            continue
        client.download(dict(source_uri=destination+name, size=size, etag=''), path)


def command(args, session):
    return [sys.executable, 'depth', '--source', str(session), '--model-root', str(args.model_root),
            '--ckpt', str(args.ckpt), '--engine', 'throughput', '--num-frames', 'auto',
            '--batch-size', str(args.batch_size), '--depth-format', 'hevc12',
            '--hevc12-preset', 'fast', '--ffmpeg', args.ffmpeg]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(value == '--source' or value.startswith('--source=') for value in argv):
        # The L1 camera names match Jianzhi; its CLI delegates inference to base.
        from roboego_hand_vis.lerobot_fast_foundation_depth.L1.run import main as local_main
        if not any(value == '--hevc12-preset' or value.startswith('--hevc12-preset=') for value in argv):
            argv += ['--hevc12-preset', 'fast']
        return local_main(argv)

    p = argparse.ArgumentParser(description=__doc__, epilog='Local mode: --source PATH [L1 depth options, including --dry-run].')
    p.add_argument('--state-root', type=Path, required=True, help='Jianzhi pipeline state-root with batch_*/records/*.json')
    p.add_argument('--download-root', type=Path, required=True, help='Keep downloaded datasets and depth here')
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--ks3-tool', required=True)
    p.add_argument('--ks3-config', required=True)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--ffprobe', default='ffprobe')
    p.add_argument('--gpus', default='0')
    p.add_argument('--depth-workers-per-gpu', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1, help='Stereo frame pairs per GPU inference call')
    p.add_argument('--first-batch', type=int, default=1)
    p.add_argument('--last-batch', type=int, default=9999999)
    p.add_argument('--poll-seconds', type=float, default=60)
    p.add_argument('--min-free-gb', type=float, default=50)
    p.add_argument('--once', action='store_true', help='Drain current indices and exit instead of watching')
    p.add_argument('--execute', action='store_true', help='Otherwise print pending count only')
    a = p.parse_args(argv)
    devices = a.gpus.split(',')
    if (any(not d.isdigit() for d in devices) or len(set(devices)) != len(devices)
            or a.depth_workers_per_gpu < 1 or a.batch_size < 1 or a.poll_seconds <= 0 or a.min_free_gb < 0
            or not 1 <= a.first_batch <= a.last_batch):
        p.error('Invalid GPU, polling, disk-space or batch options')
    if not a.state_root.is_dir():
        p.error('state-root does not exist')
    if not a.execute:
        print('Pending uploaded datasets:', sum(1 for _ in discover(a.state_root, a.first_batch, a.last_batch)))
        return 0
    for path in (a.ckpt, a.ckpt.with_name('cfg.yaml')):
        if not path.is_file():
            p.error('Missing model file: '+str(path))
    if not a.model_root.is_dir() or not shutil.which(a.ffmpeg):
        p.error('model-root or ffmpeg unavailable')
    a.download_root = a.download_root.resolve()
    a.download_root.mkdir(parents=True, exist_ok=True)
    lock = a.state_root/'depth_runner.lock'
    try:
        lock.open('x').close()
    except FileExistsError:
        p.error('depth_runner.lock exists; check for another depth coordinator')
    slots = devices*a.depth_workers_per_gpu
    queue = Queue(maxsize=len(slots)*2)
    stop, guard = Event(), Lock()
    seen, failures = set(), []
    logs = a.download_root/'logs'
    logs.mkdir(exist_ok=True)

    def lane(device):
        worker = None
        client = KS3(a.ks3_tool, a.ks3_config)
        try:
            while not stop.is_set():
                try:
                    task = queue.get(timeout=1)
                except Empty:
                    continue
                if task is None:
                    queue.task_done()
                    break
                batch, row = task
                uri = row['destination_uri']
                # IDs are derived from the URI, never use remote path fragments as local names.
                import hashlib
                session = a.download_root/batch/hashlib.sha256(uri.encode()).hexdigest()
                record_path = a.state_root/batch/'depth_records'/(session.name+'.json')
                record = dict(source_uri=uri, session=str(session), gpu=device)
                def write(status, **extra):
                    record.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
                    atomic_json(record_path, record)
                try:
                    size = sum(row['upload']['file_sizes'].values())
                    if shutil.disk_usage(a.download_root).free < size+a.min_free_gb*1024**3:
                        raise RuntimeError('Insufficient free space; downloaded data is retained')
                    write('downloading')
                    download_dataset(client, row, session)
                    write('processing')
                    if worker is None:
                        worker = Worker('depth', device, logs)
                    log = logs/batch/(session.name+'.log')
                    log.parent.mkdir(exist_ok=True)
                    if worker.execute(command(a, session), log) != 0:
                        raise RuntimeError('Depth worker failed; see '+str(log))
                    from roboego_hand_vis.lerobot_fast_foundation_depth.base.compute import plan_session
                    from roboego_hand_vis.lerobot_fast_foundation_depth.base.reuse import existing_valid
                    plans = plan_session(session)
                    # Keep metadata checks without decoding the entire output video again.
                    if not plans or not all(existing_valid(plan, verify_video=False) for plan in plans):
                        raise RuntimeError('Depth output validation failed')
                    write('done', log=str(log))
                    with guard:
                        done_file = a.state_root/batch/'depth_done.json'
                        atomic_json(done_file, sorted(set(load(done_file, []))|{uri}))
                    print('DONE', batch, uri, flush=True)
                except Exception as exc:
                    write('failed', error=str(exc))
                    with guard:
                        failures.append(uri)
                    print('FAILED', uri, str(exc), flush=True)
                    if worker is not None and worker.process.poll() is not None:
                        worker.close()
                        worker = None
                finally:
                    queue.task_done()
        finally:
            if worker is not None:
                worker.close()

    try:
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = [pool.submit(lane, d) for d in slots]
            try:
                while not stop.is_set():
                    added = 0
                    for batch, row in discover(a.state_root, a.first_batch, a.last_batch):
                        uri = row['destination_uri']
                        if uri in seen:
                            continue
                        queue.put((batch, row))
                        seen.add(uri)
                        added += 1
                    print('Scan: queued', added, 'new datasets', flush=True)
                    if a.once:
                        queue.join()
                        break
                    stop.wait(a.poll_seconds)
            finally:
                stop.set()
            for future in futures:
                future.result()
    finally:
        lock.unlink(missing_ok=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
