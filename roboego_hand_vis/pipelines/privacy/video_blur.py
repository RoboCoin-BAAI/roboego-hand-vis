"""Session adapter for the project privacy-blur batch runtime.

This module deliberately imports only the standard library. Models and inference
libraries belong to the configured child interpreter, never to the Kedro process.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAMERAS = ('left', 'right', 'bleft', 'bright')
STAGE = 'video_privacy_blur'
DEFAULTS = {
    'python': 'python', 'camera_streams': 'auto',
    'face_weights': 'conf/models/yolov8n-face.pt',
    'device': 'auto', 'conf': .35, 'imgsz': 960, 'scale': 1.35,
    'method': 'gaussian', 'blur_strength': 1, 'track_gap': 0,
    'track_scale': 1.25, 'jobs': 1, 'threads': 2, 'retries': 1, 'timeout': None,
}


def _path(value: Any, key: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f'{STAGE}.{key} is required when enabled')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _number(config: dict, key: str, minimum: float, *, integer=False, inclusive=True):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{STAGE}.{key} must be a finite number')
    if (value < minimum if inclusive else value <= minimum) or (integer and int(value) != value):
        raise ValueError(f'{STAGE}.{key} has an invalid value: {value}')
    config[key] = int(value) if integer else value


def _prepare(config: dict) -> tuple[dict, Path, Path, list[str]]:
    cfg = {**DEFAULTS, **config}
    session = _path(cfg.get('session_dir'), 'session_dir')
    videos = session / 'videos'
    if not videos.is_dir() or videos.is_symlink():
        raise ValueError(f'{STAGE}: session must contain a regular videos directory: {videos}')
    selection = cfg['camera_streams']
    if selection == 'auto':
        cameras = [name for name in CAMERAS if (videos / f'{name}.mp4').exists()]
    else:
        cameras = selection.split('|') if isinstance(selection, str) else selection
    if (not isinstance(cameras, list) or not cameras
            or any(not isinstance(name, str) or name not in CAMERAS for name in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError(f'{STAGE}.camera_streams: select unique original cameras from {CAMERAS}')
    for name in cameras:
        source = videos / f'{name}.mp4'
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f'{STAGE}: missing or symlink camera video: {source}')
    output = _path(cfg['output_dir'], 'output_dir') if cfg.get('output_dir') else session / 'privacy' / STAGE
    output = output.resolve()
    if output == videos or output in videos.parents or videos in output.parents:
        raise ValueError(f'{STAGE}: input and output directories must not overlap')
    for key, allowed in [('device', ('auto', 'cpu', 'cuda', 'mps')),
                         ('method', ('gaussian', 'pixelate', 'egoblur'))]:
        if cfg[key] not in allowed:
            raise ValueError(f'{STAGE}.{key} must be one of {allowed}')
    for key, minimum in [('conf', .25), ('scale', 1), ('track_scale', 1)]:
        _number(cfg, key, minimum)
    if cfg['conf'] > 1:
        raise ValueError(f'{STAGE}.conf must be <= 1')
    _number(cfg, 'blur_strength', 0, inclusive=False)
    for key, minimum in [('imgsz', 1), ('threads', 1), ('track_gap', 0), ('retries', 0)]:
        _number(cfg, key, minimum, integer=True)
    if cfg['jobs'] != 'auto':
        if isinstance(cfg['jobs'], str) and cfg['jobs'].isdigit():
            cfg['jobs'] = int(cfg['jobs'])
        _number(cfg, 'jobs', 1, integer=True)
    if cfg['timeout'] is not None:
        _number(cfg, 'timeout', 0, inclusive=False)
    weights = _path(cfg['face_weights'], 'face_weights')
    if not weights.is_file():
        raise FileNotFoundError(f'{STAGE}: weights not found: {weights}')
    cfg['face_weights'] = str(weights)
    python = cfg['python']
    if not isinstance(python, str) or not python.strip():
        raise ValueError(f'{STAGE}.python must be an executable path or name')
    if '/' in python:
        executable = _path(python, 'python')
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f'{STAGE}: interpreter not executable: {executable}')
        # Preserve the venv path: resolving its symlink would bypass the environment.
        expanded = Path(python).expanduser()
        cfg['python'] = str(expanded if expanded.is_absolute() else PROJECT_ROOT / expanded)
    else:
        cfg['python'] = shutil.which(python)
        if cfg['python'] is None:
            raise FileNotFoundError(f'{STAGE}: interpreter not found: {python}')
    for binary in ('ffmpeg', 'ffprobe'):
        if not shutil.which(binary):
            raise FileNotFoundError(f'{STAGE}: {binary} is required on PATH')
    return cfg, session, output, cameras


@contextmanager
def _session_lock(output: Path):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.session.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'{STAGE}: another session run owns {output}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run_batch(args: list[str], *, env: dict, cwd: Path, log_path: Path) -> int:
    """Forward interruption to the batch supervisor so it cleans up its workers."""
    def interrupt(signum, frame):
        raise KeyboardInterrupt()

    previous = signal.getsignal(signal.SIGTERM)
    install_handler = threading.current_thread() is threading.main_thread()
    if install_handler:
        signal.signal(signal.SIGTERM, interrupt)
    try:
        with log_path.open('a', encoding='utf-8') as log:
            process = subprocess.Popen(args, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                return process.wait()
            except BaseException:
                # Batch worker processes own separate process groups. Give the batch
                # supervisor time to terminate/reap them before returning to Kedro.
                # Include resource-calibration children started before the batch
                # supervisor installs its own signal handler.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
    finally:
        if install_handler:
            signal.signal(signal.SIGTERM, previous)


def _read_summary(path: Path, expected: int) -> dict:
    try:
        summary = json.loads(path.read_text(encoding='utf-8'))
        counts = summary['counts']
        values = [counts[key] for key in ('completed', 'skipped', 'failed')]
        if (any(type(value) is not int or value < 0 for value in values)
                or sum(values) != expected or not isinstance(summary['failures'], list)
                or len(summary['failures']) != counts['failed']):
            raise ValueError('inconsistent counts')
        return {'counts': counts, 'failures': summary['failures']}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f'{STAGE}: missing or invalid batch report: {path}') from exc


def run_video_privacy_blur_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    enabled = config.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError(f'{STAGE}.enabled must be a boolean')
    if not enabled:
        return {'stage': STAGE, 'status': 'skipped', 'reason': 'disabled'}
    cfg, session, output, cameras = _prepare(config)
    args = [cfg['python'], '-m', 'roboego_hand_vis.video_privacy_blur.batch', str(session / 'videos'), '--output-dir', str(output)]
    for name in cameras:
        args.extend(['--video', f'{name}.mp4'])
    for key in ('face_weights', 'device', 'conf', 'imgsz', 'scale', 'method', 'blur_strength',
                'track_gap', 'track_scale', 'jobs', 'threads', 'retries', 'timeout'):
        if cfg[key] is not None:
            args.extend(['--' + key.replace('_', '-'), str(cfg[key])])
    env = os.environ.copy()
    # Pin imports for both the supervisor and all Python children, even if another
    # roboego_hand_vis package is installed in the configured runtime.
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    with _session_lock(output):
        summary_path = output / '.batch/summary.json'
        summary_path.unlink(missing_ok=True)
        report_path = output / 'session_report.json'
        report = {
            'stage': STAGE, 'status': 'running', 'session_dir': str(session), 'output_dir': str(output),
            'started_at': datetime.now(timezone.utc).isoformat(), 'command': args,
            'log_path': str(output / 'run.log'), 'batch_logs': str(output / '.batch/logs'),
            'batch_summary': str(summary_path),
            'cameras': {name: {'input': str(session / 'videos' / f'{name}.mp4'),
                               'output': str(output / f'{name}_blurred.mp4')}
                        for name in cameras},
            'counts': None, 'failures': [],
        }
        _write_report(report_path, report)
        try:
            code = _run_batch(args, env=env, cwd=PROJECT_ROOT, log_path=output / 'run.log')
            report['returncode'] = code
            if summary_path.exists() or code == 0:
                report.update(_read_summary(summary_path, len(cameras)))
            if code != 0 or report['counts']['failed']:
                raise RuntimeError(f'{STAGE}: batch failed (exit {code}); see {output / "run.log"}')
            for camera in report['cameras'].values():
                result = Path(camera['output'])
                if not result.is_file() or result.is_symlink() or result.stat().st_size == 0:
                    raise RuntimeError(f'{STAGE}: batch report references missing output: {result}')
            report['status'] = 'complete'
        except BaseException as exc:
            report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            _write_report(report_path, report)
        return report
