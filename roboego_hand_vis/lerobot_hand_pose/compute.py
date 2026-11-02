"""Run external hand reconstruction without importing its GPU runtime here."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from queue import Queue, Empty
from threading import Thread
import time
from uuid import uuid4


def run_logged(command, *, cwd, env, log_path, label):
    """Stream unbuffered model output to both the Job console and task log."""
    env = dict(env, PYTHONUNBUFFERED='1')
    with log_path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        print(f'[{label}] model PID={child.pid}; log={log_path}', flush=True)
        lines = Queue()
        def reader():
            try:
                for line in child.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        thread = Thread(target=reader, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            while True:
                try:
                    line = lines.get(timeout=20)
                except Empty:
                    print(f'[{label}] model still running; elapsed={time.monotonic()-started:.0f}s; PID={child.pid}', flush=True)
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                print(f'[{label}] {line.rstrip()}', flush=True)
            return subprocess.CompletedProcess(command, child.wait())
        except BaseException:
            child.terminate()
            child.wait()
            raise
        finally:
            child.stdout.close()
            thread.join(timeout=2)


def identity(video, timestamps):
    stat = video.stat()
    return dict(video_size=stat.st_size, video_mtime_ns=stat.st_mtime_ns,
                timestamp_sha256=hashlib.sha256(timestamps.read_bytes()).hexdigest())


def vio_succeeded(session, episode):
    """Use the current published pose's report, including selected ZUPT runs."""
    try:
        provenance = json.loads((episode / 'relative_head_pose_provenance.json').read_text(encoding='utf-8'))
        report_path = (session / provenance['report']).resolve()
        if not report_path.is_relative_to(session) or not provenance.get('published_to_lerobot'):
            return False
        report = json.loads(report_path.read_text(encoding='utf-8'))
        return (report.get('status') == 'complete' and report.get('initialized') is True
                and report.get('quality_status') == 'passed_basic_checks'
                and report.get('quality', {}).get('status') == 'passed_basic_checks'
                and provenance.get('quality', {}).get('status') == 'passed_basic_checks'
                and (episode / 'observation.relative_head_pose.parquet').is_file()
                and (episode / 'observation.relative_head_pose_status.parquet').is_file())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def plans(session, level, vio_success_only=False):
    session = Path(session).resolve()
    camera = 'head_left_rectified' if level == 'L1' else 'headband_left_rgb_rectified'
    episodes = sorted(p for p in (session / 'data').glob('chunk-*/episode_*') if p.is_dir())
    if not episodes:
        raise ValueError('No LeRobot episodes')
    result = []
    for episode in episodes:
        if vio_success_only and not vio_succeeded(session, episode):
            print(f'[hand-skip] {session.name}/{episode.name}: no current QC-passed VIO result', flush=True)
            continue
        feature = f'observation.images.{camera}'
        video = session / 'videos' / episode.parent.name / feature / f'{episode.name}.mp4'
        timestamps = episode / f'{feature}.parquet'
        if not video.is_file() or not timestamps.is_file():
            raise ValueError(f'Missing hand input for {episode}: {feature}')
        result.append(dict(session=session, episode=episode, video=video, timestamps=timestamps,
                           feature=feature, relative=Path('hand') / episode.parent.name / episode.name))
    return result


def validate_prediction(output, timestamps):
    import numpy as np
    import pyarrow.parquet as pq
    count = pq.read_metadata(timestamps).num_rows
    with np.load(output / 'prediction.npz', allow_pickle=False) as arrays:
        if count < 1 or not np.array_equal(arrays['frame_indices'], np.arange(count)):
            raise ValueError('Hand prediction frame indices do not match source Parquet rows')
        if len(arrays['joints_3d']) != count:
            raise ValueError('Hand joint count mismatch')
    metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
    expected = metadata.get('prediction_sha256')
    if expected and hashlib.sha256((output / 'prediction.npz').read_bytes()).hexdigest() != expected:
        raise ValueError('Hand prediction checksum mismatch')


def existing_valid(plan, model_root, precision='auto', window_batch_size=1, resize_mode='long_edge_pad'):
    output = plan['session'] / plan['relative']
    try:
        receipt = json.loads((output / 'lerobot_hand.json').read_text(encoding='utf-8'))
        expected = dict(precision=precision, window_batch_size=window_batch_size, resize_mode=resize_mode,
                        overlay=False, model_root=str(Path(model_root).resolve()))
        if receipt['source'] != identity(plan['video'], plan['timestamps']) or receipt['settings'] != expected:
            return False
        if receipt.get('camera_feature') != plan['feature']:
            return False
        validate_prediction(output, plan['timestamps'])
        return True
    except (OSError, ValueError, KeyError):
        return False


def process(session, level, *, model_root, python, gpu='0', precision='auto',
            window_batch_size=1, resize_mode='long_edge_pad', skip_existing=True,
            dry_run=False, ffmpeg_dir=None, vio_success_only=False, selected_plans=None, inference=None):
    from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import publish_files
    entry = Path(model_root).resolve() / 'infer.py'
    if not entry.is_file():
        raise ValueError(f'Missing hand model entry: {entry}')
    results = []
    for plan in (selected_plans if selected_plans is not None else plans(session, level, vio_success_only=vio_success_only)):
        if vio_success_only and not vio_succeeded(plan['session'], plan['episode']):
            raise ValueError('VIO eligibility changed since selection')
        root = plan['session']
        output = root / plan['relative']
        settings = dict(precision=precision, window_batch_size=window_batch_size, resize_mode=resize_mode,
                        overlay=False, model_root=str(entry.parent))
        source = identity(plan['video'], plan['timestamps'])
        receipt = dict(source=source, settings=settings, camera_feature=plan['feature'],
                       timestamp_lookup='source_timestamps.parquet row indexed by original frame number',
                       coordinate_system='model camera frame, OpenCV axes, meters; metric accuracy not verified',
                       calibrated_intrinsics_used_for_3d=False)
        if dry_run:
            results.append(dict(video=str(plan['video']), output=str(output), settings=settings))
            continue
        if skip_existing:
            try:
                if json.loads((output / 'lerobot_hand.json').read_text(encoding='utf-8')) == receipt:
                    validate_prediction(output, plan['timestamps'])
                    results.append(dict(output=str(output), skipped_existing=True))
                    continue
            except (OSError, ValueError, KeyError):
                pass
        staging = root / f'.hand.partial-{uuid4().hex}'
        destination = staging / plan['relative']
        command = [str(python), '-u', str(entry), str(plan['video']), '--output', str(destination),
                   '--device', 'cuda:0', '--precision', precision, '--resize-mode', resize_mode,
                   '--window-batch-size', str(window_batch_size), '--max-frames', '0', '--no-overlay']
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if ffmpeg_dir:
            env['PATH'] = str(ffmpeg_dir) + os.pathsep + env.get('PATH', '')
        staging.mkdir()
        print(f'[hand] {root.name}/{plan["episode"].name}: GPU={gpu}', flush=True)
        # Failed staging and logs remain for diagnosis; published results are untouched.
        if inference is None:
            result = run_logged(command, cwd=entry.parent, env=env, log_path=staging / 'model.log',
                                label=f'hand GPU={gpu} {root.name}/{plan["episode"].name}')
        else:
            result = inference(plan['video'], destination, staging / 'model.log')
        if result.returncode:
            raise RuntimeError(f'Hand model exited {result.returncode}; see {staging / "model.log"}')
        validate_prediction(destination, plan['timestamps'])
        if source != identity(plan['video'], plan['timestamps']):
            raise ValueError('Hand inputs changed during inference')
        shutil.copy2(plan['timestamps'], destination / 'source_timestamps.parquet')
        (destination / 'lerobot_hand.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        shutil.move(str(staging / 'model.log'), str(destination / 'model.log'))
        publish_files(staging, root)
        results.append(dict(output=str(output), skipped_existing=False))
    return results
