#!/usr/bin/env python3
"""In-place DA3 depth orchestration. Inference comes from RoboCOIN-with-Depth.

Only depth MP4, depth timestamp parquet and meta/info.json are published.
Everything else lives in --work-dir (must be outside data, on the same filesystem).
One spawned process/model per GPU; sessions are dynamically assigned to workers.
Do not run another writer against these sessions concurrently.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from fractions import Fraction
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def stamp(path):
    st = Path(path).stat()
    return [st.st_size, st.st_mtime_ns]


def probe(path, count=False):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0']
    if count:
        cmd += ['-count_frames']
    cmd += ['-show_entries',
            'stream=width,height,avg_frame_rate,nb_read_frames,pix_fmt,codec_name',
            '-of', 'json', str(path)]
    return json.loads(subprocess.check_output(cmd, text=True))['streams'][0]


def sessions(root, pattern, camera, first=None, last=None, session=None):
    for path in sorted(root.glob(pattern)):
        if not path.is_dir() or path.is_symlink():
            continue
        if session and path.name != session:
            continue
        if first is not None or last is not None:
            batch = path.parent.name
            if not batch.startswith('batch_') or not batch[6:].isdigit():
                continue
            number = int(batch[6:])
            if first is not None and number < first:
                continue
            if last is not None and number > last:
                continue
        videos = sorted((path / 'videos').glob(f'chunk-*/{camera}/episode_*.mp4'))
        if videos:
            yield path, videos


def rollback(folder):
    """Recover interrupted publication before workers start. Backups are external."""
    journal = folder / 'transaction.json'
    if not journal.exists():
        return
    for item in json.loads(journal.read_text(encoding='utf-8')):
        target = Path(item['target'])
        if item['existed']:
            # Copy to an external staging file, then atomic replacement.
            restore = folder / 'restore'
            shutil.copy2(item['backup'], restore)
            os.replace(restore, target)
        else:
            target.unlink(missing_ok=True)
    journal.unlink()


def publish(folder, pairs):
    """Atomic per-file replacement, with rollback/recovery for the three-file set."""
    items = []
    for index, (_, target) in enumerate(pairs):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = folder / f'backup-{index}'
        existed = target.exists()
        if existed:
            shutil.copy2(target, backup)
        items.append(dict(target=str(target), backup=str(backup), existed=existed))
    write_json(folder / 'transaction.json', items)
    try:
        for source, target in pairs:
            os.replace(source, target)
    except BaseException:
        rollback(folder)
        raise
    (folder / 'transaction.json').unlink()
    for item in items:
        Path(item['backup']).unlink(missing_ok=True)


def update_info(info, args, stream):
    info = copy.deepcopy(info)
    source = info['features'][args.camera]
    feature = copy.deepcopy(source)
    feature.update(dtype='video', shape=[stream['height'], stream['width'], 1],
                   names=['h', 'w', 'c'], fps=float(Fraction(stream['avg_frame_rate'])))
    # Replace old stereo storage/mask metadata rather than retaining stale references.
    feature['info'] = {
        'video.fps': feature['fps'], 'video.codec': 'hevc',
        'video.height': stream['height'], 'video.width': stream['width'],
        'video.channels': 1, 'video.pix_fmt': 'gray12le',
        'video.is_depth_map': True, 'has_audio': False,
        'depth.unit': 'mm', 'depth.scale': 1,
        'depth.decode': 'depth_mm = decoded_pixel * depth.scale',
        'depth.invalid_value': 0, 'depth.max_mm': 4095,
        'depth.method': 'da3_monocular', 'depth.model': args.model,
        'depth.source': args.camera, 'depth.crf': args.crf,
        'depth.lossless': False, 'depth.process_res': args.process_res,
        'depth.chunk_size': args.chunk_size, 'depth.overlap': args.overlap,
    }
    info['features'][args.depth_feature] = feature
    return info


def decoded_chunks(video, width, height, count, chunk_size, overlap):
    """Decode at most one inference chunk, retaining only its overlap for reuse."""
    import numpy as np
    from src.depth_repair.da3 import _make_chunks_streaming

    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    frames, offset, read_count = [], 0, 0
    frame_bytes = width * height * 3
    try:
        for start, end in _make_chunks_streaming(count, chunk_size, overlap):
            if start > offset:
                del frames[:start - offset]
                offset = start
            while read_count < end:
                data = proc.stdout.read(frame_bytes)
                if len(data) != frame_bytes:
                    raise ValueError(f'RGB decode truncated at frame {read_count}/{count}')
                frames.append(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3))
                read_count += 1
            # The decoder mutates its rolling list on advance; queued chunks
            # need independent lists (pixel buffers remain shared/read-only).
            yield start, end, list(frames)
        if proc.stdout.read(1):
            raise ValueError('RGB has more frames than timestamps')
        if proc.wait() != 0:
            raise RuntimeError('RGB decoder failed')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def align_overlap(current, previous, overlap):
    """Same low-resolution scale/shift fit and blend as upstream _infer_streaming."""
    import numpy as np
    before, after = previous, current[:overlap]
    mask = (before > 0) & np.isfinite(before) & (after > 0) & np.isfinite(after)
    if mask.sum() >= 100:
        p, t = after[mask].ravel(), before[mask].ravel()
        matrix = np.stack([p, np.ones_like(p)], axis=1)
        scale, shift = np.linalg.lstsq(matrix, t, rcond=None)[0]
        scale = float(np.clip(scale, 0.5, 2.0))
        current = np.maximum(current * scale + shift, 0)
        weight = np.linspace(0, 1, overlap, dtype=np.float32).reshape(overlap, 1, 1)
        current[:overlap] = (1 - weight) * before + weight * current[:overlap]
    return current


def depth_frames(model, chunks, count, width, height, chunk_size, overlap, blocks=False):
    """Reuse DA3 forward and original alignment, retaining no whole-video arrays."""
    import cv2
    import numpy as np
    import torch

    previous = None
    streaming = count > chunk_size and overlap > 0
    for start, end, rgb in chunks:
        began = time.monotonic()
        with torch.inference_mode():
            if not streaming:
                # Preserve upstream simple-path resize and OOM retry behavior.
                output = model.repair_frames(rgb, [])
                if blocks:
                    yield output
                else:
                    yield from output
                del output
            else:
                # These are the identical calls used by DA3DepthRepair._infer_streaming.
                network = model._model
                cpu, _, _ = network._preprocess_inputs(
                    rgb, None, None, model._process_res, 'upper_bound_resize')
                imgs, _, _ = network._prepare_model_inputs(cpu, None, None)
                raw = network.forward(imgs, None, None, export_feat_layers=[])
                pred = network._convert_to_prediction(raw)
                current = np.array(pred.depth, dtype=np.float32, copy=True)
                del cpu, imgs, raw, pred
                if len(current) != end - start:
                    raise ValueError('DA3 returned the wrong chunk frame count')
                if previous is not None:
                    current = align_overlap(current, previous, overlap)
                emit = len(current) if end == count else len(current) - overlap
                previous = current[-overlap:].copy()
                # Match upstream order: multiply in float32 before cv2 resize.
                def resized():
                    for index in range(emit):
                        yield cv2.resize(current[index] * 1000.0, (width, height),
                                         interpolation=cv2.INTER_LINEAR)
                if blocks:
                    yield list(resized())
                else:
                    yield from resized()
                del current
        print(f'Chunk {start}:{end}/{count} completed in '
              f'{time.monotonic() - began:.1f}s (including encode backpressure)', flush=True)


class PipelineStalled(BaseException):
    """A background producer cannot be reused safely; terminate this GPU worker."""


@contextlib.contextmanager
def prefetched(source, capacity, name):
    """Bounded producer queue with ordered results, exceptions and cancellation."""
    pending = queue.Queue(maxsize=capacity)
    stopped = threading.Event()
    end = object()

    def send(value):
        while not stopped.is_set():
            try:
                pending.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        try:
            for value in source:
                if stopped.is_set():
                    break
                send((True, value))
            send((True, end))
        except BaseException as exc:
            send((False, exc))
        finally:
            source.close()

    def consume():
        while True:
            valid, value = pending.get()
            if not valid:
                raise value
            if value is end:
                return
            yield value

    thread = threading.Thread(target=produce, name=name, daemon=True)
    thread.start()
    consumer = consume()
    try:
        yield consumer
    finally:
        stopped.set()
        consumer.close()
        thread.join(timeout=120)
        if thread.is_alive():
            raise PipelineStalled(f'{name} failed to stop; refusing to reuse GPU worker')


def flatten_blocks(blocks):
    for block in blocks:
        yield from block


def validate_video(path, count, width, height):
    """Inspect headers, then decode once to check both errors and frame count."""
    stream = probe(path)
    if (stream['width'] != width or stream['height'] != height
            or stream['pix_fmt'] != 'gray12le' or stream['codec_name'] != 'hevc'):
        raise ValueError('Encoded depth format validation failed')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
        '-map', '0:v:0', '-progress', 'pipe:1', '-nostats', '-f', 'null', '-'],
        stdout=subprocess.PIPE, text=True)
    frames = None
    try:
        for line in proc.stdout:
            if line.startswith('frame='):
                frames = int(line.split('=', 1)[1])
        if proc.wait() != 0 or frames != count:
            raise ValueError(f'Depth decode/count validation failed: {frames} != {count}')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def encode_stream(encoder, frames, width, height, destination, count):
    """Use upstream FFmpeg command and pixel conversion without copying a video list."""
    import numpy as np
    proc = subprocess.Popen(encoder._build_cmd(width, height, str(destination)),
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    emitted = 0
    try:
        for frame in frames:
            if frame.shape != (height, width) or not np.isfinite(frame).all():
                raise ValueError('Invalid DA3 output frame')
            proc.stdin.write(np.clip(frame, 0, 4095).astype(np.uint16).tobytes())
            emitted += 1
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('Depth encoder failed; inspect GPU log for FFmpeg errors')
        if emitted != count:
            raise ValueError(f'Depth frame count mismatch: {emitted} != {count}')
    finally:
        frames.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if not proc.stdin.closed:
            with contextlib.suppress(BrokenPipeError, OSError):
                proc.stdin.close()


def process_video(args, model, video, session, work, signature):
    import numpy as np
    import pyarrow.parquet as pq
    from src.depth_encoder import DepthVideoEncoder

    chunk = video.parent.parent.name
    source_time = session / 'data' / chunk / video.stem / (args.camera + '.parquet')
    output = session / 'videos' / chunk / args.depth_feature / video.name
    output_time = session / 'data' / chunk / video.stem / (args.depth_feature + '.parquet')
    info_path = session / 'meta' / 'info.json'
    for target in (output, output_time, info_path):
        if not target.resolve().is_relative_to(session.resolve()):
            raise ValueError(f'Output escapes session via symlink: {target}')
        if target.is_symlink():
            raise ValueError(f'Refusing symlink output: {target}')
        ancestor = target.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if ancestor.stat().st_dev != work.stat().st_dev:
            raise ValueError(f'Output and work-dir are on different filesystems: {target}')
    task_id = hashlib.sha256(str(video).encode()).hexdigest()
    folder = work / 'tasks' / task_id
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / 'done.json'
    fingerprint = dict(source=stamp(video), timestamps=stamp(source_time), config=signature)
    if record.exists() and not args.force:
        saved = json.loads(record.read_text(encoding='utf-8'))
        if (saved.get('input') == fingerprint and output.exists() and output_time.exists()
                and saved.get('video') == stamp(output)
                and saved.get('timestamps') == stamp(output_time)
                and saved.get('feature') == json.loads(info_path.read_text(encoding='utf-8'))
                .get('features', {}).get(args.depth_feature)):
            return 'skipped'
    table = pq.read_table(source_time)
    if 'timestamp' not in table.column_names or table.num_rows == 0:
        raise ValueError(f'Missing/empty timestamp column: {source_time}')
    times = table['timestamp'].to_numpy()
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError(f'Timestamps must be finite and strictly increasing: {source_time}')
    stream = probe(video)
    fps = Fraction(stream['avg_frame_rate'])
    if fps <= 0:
        raise ValueError(f'Invalid FPS: {video}')

    class Encoder(DepthVideoEncoder):
        def _build_cmd(self, w, h, save_path):
            cmd = super()._build_cmd(w, h, save_path)
            # Upstream uses FPS as GOP length; GOP must be integer for fractional FPS.
            for option in ('-g', '-keyint_min'):
                cmd[cmd.index(option) + 1] = str(max(1, round(float(fps))))
            return cmd

    staged_video = folder / 'depth.mp4'
    chunks = decoded_chunks(video, stream['width'], stream['height'], table.num_rows,
                            args.chunk_size, args.overlap)
    try:
        encoder = Encoder(fps=str(fps), crf=args.crf, preset=args.preset)
        with prefetched(chunks, 1, 'rgb-prefetch') as rgb:
            blocks = depth_frames(model, rgb, table.num_rows, stream['width'], stream['height'],
                                  args.chunk_size, args.overlap, blocks=True)
            with prefetched(blocks, 2, 'depth-prefetch') as depths:
                encode_stream(encoder, flatten_blocks(depths), stream['width'], stream['height'],
                              staged_video, table.num_rows)
        print('Inference/encoding complete; validating depth video', flush=True)
        validate_video(staged_video, table.num_rows, stream['width'], stream['height'])
        staged_time = folder / 'timestamps.parquet'
        shutil.copy2(source_time, staged_time)
        if not pq.read_table(staged_time).equals(table):
            raise ValueError('Timestamp verification failed')
        info = update_info(json.loads(info_path.read_text(encoding='utf-8')), args, stream)
        info['total_videos'] = sum(1 for _ in (session / 'videos').glob(
            'chunk-*/observation.images.*/episode_*.mp4')) + int(not output.exists())
        staged_info = folder / 'info.json'
        write_json(staged_info, info)
        if fingerprint != dict(source=stamp(video), timestamps=stamp(source_time), config=signature):
            raise RuntimeError('Input changed during inference')
        publish(folder, [(staged_video, output), (staged_time, output_time),
                         (staged_info, info_path)])
        write_json(record, dict(input=fingerprint, video=stamp(output),
                                timestamps=stamp(output_time),
                                feature=info['features'][args.depth_feature]))
        return 'success'
    finally:
        chunks.close()
        for name in ('depth.mp4', 'timestamps.parquet', 'info.json'):
            (folder / name).unlink(missing_ok=True)


def worker(gpu, args, tasks, results, signature):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    sys.path.insert(0, str(Path(args.robocoin_root)))
    sys.path.insert(0, str(Path(args.robocoin_root) / 'third_party/Depth-Anything-3/src'))
    log = Path(args.work_dir) / f'gpu-{gpu}.log'
    with log.open('a', encoding='utf-8', buffering=1) as output:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            # Also route native library and subprocess output outside the data tree.
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            try:
                # Check timestamp dependencies before allocating/loading the large model.
                import pyarrow.parquet
                import torch
                from src.depth_repair import create_depth_repair
                if not torch.cuda.is_available():
                    raise RuntimeError(f'CUDA unavailable on GPU {gpu}')
                torch.set_num_threads(2)
                model = create_depth_repair('da3', model_id=args.model,
                    process_res=args.process_res, chunk_size=args.chunk_size,
                    overlap=args.overlap, device='cuda:0')
                model._load_model()  # Eager load once; keep this object for the worker lifetime.
                print(f'GPU {gpu}: model loaded', flush=True)
                while True:
                    task = tasks.get()
                    if task is None:
                        return
                    session, videos = task
                    for video in videos:
                        try:
                            status = process_video(args, model, Path(video), Path(session),
                                                   Path(args.work_dir), signature)
                            results.put((status, video))
                        except Exception as exc:
                            traceback.print_exc()
                            results.put(('failed', f'{video}\n  {type(exc).__name__}: {exc}'))
                            torch.cuda.empty_cache()
            except BaseException:
                traceback.print_exc()
                raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--work-dir', required=True, type=Path)
    p.add_argument('--robocoin-root', type=Path,
                   default=Path(__file__).resolve().parents[2] / 'RoboCOIN-with-Depth')
    p.add_argument('--model', help='Local model directory (defaults to robocoin-root/models/...)')
    p.add_argument('--gpus', default='0', help='Visible CUDA device IDs, e.g. 0,1,2,3')
    p.add_argument('--session-glob', default='batch_*/*')
    p.add_argument('--session')
    p.add_argument('--first-batch', type=int)
    p.add_argument('--last-batch', type=int)
    p.add_argument('--camera', default='observation.images.head_left_rectified')
    p.add_argument('--depth-feature', default='observation.images.depth_head_left_rectified')
    p.add_argument('--process-res', type=int, default=504)
    p.add_argument('--chunk-size', type=int, default=80)
    p.add_argument('--overlap', type=int, default=20)
    p.add_argument('--crf', type=int, default=0)
    p.add_argument('--preset', choices=['fast', 'medium', 'slow', 'veryslow'], default='fast')
    p.add_argument('--limit-videos', type=int)
    p.add_argument('--force', action='store_true', help='Ignore external completion records')
    p.add_argument('--dry-run', action='store_true', help='List inputs; no writes/model imports')
    return p


def main():
    args = parser().parse_args()
    args.root = args.root.resolve()
    args.work_dir = args.work_dir.resolve()
    args.robocoin_root = args.robocoin_root.resolve()
    if not args.root.is_dir():
        raise ValueError('Input root does not exist')
    if args.work_dir == args.root or args.root in args.work_dir.parents:
        raise ValueError('--work-dir must be outside --root')
    if (args.camera == args.depth_feature or not args.camera.startswith('observation.images.')
            or not args.depth_feature.startswith('observation.images.')
            or any(c in args.camera + args.depth_feature for c in '/\\*?[]')):
        raise ValueError('Camera/depth feature must be distinct literal feature names')
    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size:
        raise ValueError('Require chunk-size > overlap >= 0')
    if args.process_res <= 0:
        raise ValueError('Process resolution must be positive')
    if not 0 <= args.crf <= 51:
        raise ValueError('CRF must be 0..51')
    if args.limit_videos is not None and args.limit_videos <= 0:
        raise ValueError('limit-videos must be positive')
    gpus = [g.strip() for g in args.gpus.split(',')]
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('--gpus must contain distinct numeric device IDs')
    jobs, count = [], 0
    for session, videos in sessions(args.root, args.session_glob, args.camera,
                                    args.first_batch, args.last_batch, args.session):
        if args.limit_videos:
            videos = videos[:args.limit_videos - count]
        if not videos:
            break
        jobs.append((str(session), [str(v) for v in videos]))
        count += len(videos)
    print(f'Sessions={len(jobs)}, videos={count}, GPUs={gpus}', flush=True)
    if args.dry_run:
        for _, videos in jobs:
            for video in videos:
                print(video)
        return
    if not jobs:
        return
    args.model = str(Path(args.model).resolve() if args.model else
                     args.robocoin_root / 'models/DA3NESTED-GIANT-LARGE-1.1')
    if not (Path(args.model) / 'model.safetensors').is_file():
        raise ValueError(f'Local model weights not found: {args.model}')
    for program in ('ffmpeg', 'ffprobe'):
        if not shutil.which(program):
            raise ValueError(f'{program} is not on PATH')
    # Import in a separate process: fail early without initializing CUDA in the parent.
    subprocess.run([sys.executable, '-c',
                    'import numpy, pyarrow.parquet, cv2, torch; '
                    'print("Dependency imports OK", flush=True)'], check=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.root.stat().st_dev != args.work_dir.stat().st_dev:
        raise ValueError('work-dir must be on the same filesystem as data for atomic replacement')
    # Linux advisory lock is outside data, and independent of work-dir choice.
    import fcntl
    lock_id = hashlib.sha256(str(args.root).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f'base-mono-depth-{lock_id}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recover previous interrupted transactions before scheduling sessions.
        for journal in (args.work_dir / 'tasks').glob('*/transaction.json'):
            rollback(journal.parent)
        signature = {k: str(getattr(args, k)) for k in (
            'root', 'model', 'camera', 'depth_feature', 'process_res', 'chunk_size',
            'overlap', 'crf', 'preset')}
        signature['weights'] = stamp(Path(args.model) / 'model.safetensors')
        signature['adapter_version'] = 2
        ctx = mp.get_context('spawn')
        tasks, results = ctx.Queue(), ctx.Queue()
        # The pending task queue can exceed pipe capacity. Never wait for its
        # feeder thread on exit after consumers have died or been terminated.
        tasks.cancel_join_thread()
        workers = [ctx.Process(target=worker, args=(g, args, tasks, results, signature))
                   for g in gpus]
        for job in jobs:
            tasks.put(job)
        for _ in workers:
            tasks.put(None)
        counts = dict(success=0, skipped=0, failed=0)
        try:
            for process in workers:
                process.start()
            while sum(counts.values()) < count:
                try:
                    status, video = results.get(timeout=1)
                    counts[status] += 1
                    print(f'{status}: {video}', flush=True)
                except queue.Empty:
                    if any(p.exitcode not in (None, 0) for p in workers):
                        raise RuntimeError('GPU worker exited unexpectedly; inspect work-dir logs')
                    if not any(p.is_alive() for p in workers):
                        raise RuntimeError('Workers exited before reporting all videos')
            for process in workers:
                process.join()
            print(json.dumps(counts), flush=True)
            if counts['failed'] or any(p.exitcode != 0 for p in workers):
                raise SystemExit(1)
        finally:
            for process in workers:
                if process.is_alive():
                    process.terminate()
            for process in workers:
                if process.pid:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join()
            tasks.close()
            results.close()


if __name__ == '__main__':
    main()
