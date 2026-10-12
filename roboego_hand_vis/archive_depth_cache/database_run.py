"""Independent L1 archive queue; keep the original standalone command."""

import argparse
import json
import shutil
import signal
import sys
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from roboego_hand_vis.common.processes import run_logged_command
from roboego_hand_vis.database.validators import resolve_session
from .cli import DEFAULT_CONFIG, config_from_yaml, load_yaml
from .compute import (DEPTH_NPY_GLOB, DEPTH_UINT16_PNG_GLOB,
                      _compression_report, _frame_idx_from_depth_path)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _inside(path, session):
    if not path.resolve().is_relative_to(session):
        raise ValueError(f"Archive path must remain inside its session: {path.name}")
    return path


def _validate(shared_root, config, limit):
    root = Path(shared_root).expanduser().resolve()
    if limit < 0 or not root.is_dir():
        raise ValueError("Need a valid shared root and nonnegative limit")
    if config.dry_run_avi:
        raise ValueError("Database archive mode does not allow dry_run_avi")
    if not (config.write_avi or config.write_hevc12):
        raise ValueError("Enable write_avi or write_hevc12 to retain archived depth")
    if config.write_hevc12 and not 0 < config.max_depth_m <= 4.095:
        raise ValueError("HEVC12 requires 0 < max_depth_m <= 4.095")
    ffmpeg = shutil.which(config.ffmpeg_bin)
    ffprobe = shutil.which(str(Path(ffmpeg).with_name('ffprobe'))) if ffmpeg else None
    ffprobe = ffprobe or shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise ValueError("Database archive mode requires ffmpeg and ffprobe")
    return root, ffprobe


def _videos(depth, config):
    result = []
    if config.write_avi:
        result.append(depth / ('depth_ffv1.mkv' if config.ffv1_container == 'mkv' else 'depth_uint16_ffv1.avi'))
    if config.write_hevc12:
        result.append(depth / 'depth_hevc12.mkv')
    if config.write_valid_mask:
        result.append(depth / 'valid_mask_ffv1.mkv')
    if config.write_preview:
        result.append(depth / 'depth_preview.mp4')
    return result


def _task(session, config, ffprobe):
    depth = _inside(session / 'depth', session).resolve()
    if not depth.is_dir():
        raise ValueError("Session depth directory is missing")
    png = _inside(depth / 'depth_uint16_png', session)
    meta = _inside(depth / 'depth_archive_meta.json', session)
    for path in _videos(depth, config):
        _inside(path, session)
    png_sources = sorted(png.glob(DEPTH_UINT16_PNG_GLOB))
    for path in png_sources:
        _inside(path, session)
    sources = sorted(depth.glob(DEPTH_NPY_GLOB))
    source_format = 'float32_npy'
    if not sources:
        sources = png_sources
        source_format = 'uint16_png'
    if not sources:
        raise ValueError("No per-frame depth cache found; archive is not rerun automatically")
    for path in sources:
        _inside(path, session)
        if not path.is_file():
            raise ValueError("Depth source must be a file")
    if source_format == 'float32_npy' and config.write_avi and not config.write_png:
        raise ValueError("write_png is required for NPY to FFV1 archive")
    indices = [_frame_idx_from_depth_path(p) for p in sources]
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate depth frame indices")
    if source_format == "float32_npy" and any(_frame_idx_from_depth_path(p) not in set(indices) for p in png_sources):
        raise ValueError("PNG cache contains frames missing from the selected NPY cache")
    for idx in indices:
        _inside(png / f"frame_{idx:06d}_depth_mm.png", session)
    annotation = _inside(session / 'annotation', session).resolve()
    annotation.mkdir(parents=True, exist_ok=True)
    for name in ('archive_depth_cache.database.yaml', 'archive_depth_cache.log'):
        _inside(annotation / name, session)
    return dict(session=session, depth_dir=depth, sources=sources, source_format=source_format,
                frame_indices=indices, meta=meta, annotation=annotation, ffprobe=ffprobe)


def _run_session(task, config):
    data = asdict(config)
    ffmpeg = data.pop('ffmpeg_bin')
    path = task['annotation'] / 'archive_depth_cache.database.yaml'
    path.write_text(yaml.safe_dump({'parameters': data, 'paths': {'ffmpeg_bin': ffmpeg}}, sort_keys=False))
    log = task['annotation'] / 'archive_depth_cache.log'
    command = [sys.executable, '-m', 'roboego_hand_vis.archive_depth_cache.run',
               '--config', str(path), '--depth_dir', str(task['depth_dir']), '--output_dir', str(task['depth_dir'])]
    run_logged_command(command, log, cwd=PROJECT_ROOT)


def _probe(path, task):
    log = _inside(task['annotation'] / f'archive_probe_{path.name}.json', task['session'])
    command = [task['ffprobe'], '-v', 'error', '-select_streams', 'v:0', '-count_frames',
               '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)]
    run_logged_command(command, log, cwd=PROJECT_ROOT)
    streams = json.loads(log.read_text()).get('streams', [])
    if len(streams) != 1:
        raise ValueError(f"Archive has no readable video stream: {path.name}")
    return int(streams[0]['nb_read_frames'])


def _check_outputs(task, config, probe):
    report = json.loads(task['meta'].read_text())
    count = len(task['frame_indices'])
    if (report.get('source_depth_dir') != str(task['depth_dir']) or report.get('output_dir') != str(task['depth_dir'])
            or report.get('frame_count') != count
            or [f.get('frame_idx') for f in report.get('frames', [])] != task['frame_indices']):
        raise RuntimeError("Archive metadata does not match the source depth frames")
    for path in _videos(task['depth_dir'], config):
        if not path.is_file() or path.stat().st_size == 0 or probe(path, task) != count:
            raise RuntimeError(f"Missing or incomplete archive video: {path.name}")
    if config.write_png and not config.delete_png:
        for idx in task['frame_indices']:
            path = task['depth_dir'] / 'depth_uint16_png' / f'frame_{idx:06d}_depth_mm.png'
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing archived PNG: {idx}")
    return report


def _cleanup(task, config, report):
    if config.delete_npy and task['source_format'] == 'float32_npy':
        for path in task['sources']:
            path.unlink()
        report['deleted_source_npy'] = True
    png = task['depth_dir'] / 'depth_uint16_png'
    if config.delete_png and png.is_dir():
        for path in png.glob(DEPTH_UINT16_PNG_GLOB):
            if _frame_idx_from_depth_path(path) in task["frame_indices"]:
                _inside(path, task['session']).unlink()
        try:
            png.rmdir()
        except OSError:
            pass
        report['deleted_uint16_png'] = not png.exists()
        if report['deleted_uint16_png']:
            report['outputs']['png_dir'] = ''
    outputs = report['outputs']
    report['compression'] = _compression_report(
        png_dir=png if png.is_dir() else None,
        ffv1_path=Path(outputs['ffv1_video']) if outputs.get('ffv1_video') else None,
        hevc12_path=Path(outputs['hevc12_video']) if outputs.get('hevc12_video') else None,
        valid_mask_path=Path(outputs['valid_mask_video']) if outputs.get('valid_mask_video') else None)
    temp = _inside(task['meta'].with_suffix('.database.tmp'), task['session'])
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    temp.replace(task['meta'])


def run_pending(service, shared_root, config, *, limit=0, runner=None, probe=None):
    root, ffprobe = _validate(shared_root, config, limit)
    runner = runner or _run_session
    probe = probe or _probe
    results = []
    while not limit or len(results) < limit:
        row = service.claim_archive_depth_cache()
        if row is None:
            break
        result = dict(id=row['id'], status='failed')
        succeeded = False
        try:
            task = _task(resolve_session(root, row['local_path']), config, ffprobe)
            runner(task, replace(config, delete_png=False, delete_npy=False))
            report = _check_outputs(task, config, probe)
            _cleanup(task, config, report)
            result.update(status='completed', frame_count=report['frame_count'], metadata_json=str(task['meta']))
            succeeded = True
        except Exception as exc:
            result['error'] = str(exc)
        except BaseException:
            if not service.finish_archive_depth_cache(row['id'], succeeded=False):
                raise RuntimeError(f"Failed to write interrupted archive status: {row['id']}")
            raise
        if not service.finish_archive_depth_cache(row['id'], succeeded=succeeded):
            raise RuntimeError(f"Failed to write archive status: {row['id']}")
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description='从 L1 数据库领取深度和手部估计均完成的数据，归档深度缓存。')
    parser.add_argument('--shared-root', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    from roboego_hand_vis.database import DataService
    url = database_url_or_error(parser)
    try:
        config = config_from_yaml(load_yaml(Path(args.config).expanduser()))
        _validate(args.shared_root, config, args.limit)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    service = DataService.from_url(url)
    def interrupted(_signum, _frame):
        raise SystemExit(143)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_pending(service, args.shared_root, config, limit=args.limit)
        print(json.dumps({'stage': 'archive_depth_cache', 'records': results}, indent=2, ensure_ascii=False))
        if any(r['status'] != 'completed' for r in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        service.database.close()


if __name__ == '__main__':
    main()
