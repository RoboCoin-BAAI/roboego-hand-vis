"""Re-encode normalized Lightwheel depth MKVs and migrate to videos/data/annotation.

Preview by default. --execute writes verified outputs; --delete-source also
removes recognized legacy files only after successful publication/verification.
Run with depth producers and the old normalize_depth.py stopped.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.depth_video import (
    HEVC12Writer, _raw_frames, read_episode_depth, cancel_active_processes,
)
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.video_storage import publish_video

FEATURE = 'observation.images.depth_head_left_rectified'
CAMERAS = {
    FEATURE: 'observation.images.head_left_rectified',
    'observation.images.depth_headband_left_rectified': 'observation.images.headband_left_rgb_rectified',
}


def discover_sources(root, first=1, last=9999999):
    """Inspect only known layout levels; never descend into data/videos/annotation."""
    root = Path(root)

    def directories(path):
        if not path.is_dir():
            return []
        # Snapshot just this directory so cleanup cannot invalidate iteration.
        with os.scandir(path) as entries:
            return sorted(Path(e.path) for e in entries if e.is_dir(follow_symlinks=False))

    def depth_sources(depth):
        for chunk in directories(depth):
            if not re.fullmatch(r'chunk-\d+', chunk.name):
                continue
            for feature in CAMERAS:
                source = chunk / feature
                if (source / 'lerobot_depth.json').is_file():
                    yield source

    if root.name == 'depth':
        yield from depth_sources(root)
        return
    if (root / 'depth').is_dir() or (root / 'data').is_dir():
        yield from depth_sources(root / 'depth')
        return
    batches = [root] if re.fullmatch(r'batch_\d+', root.name) else directories(root)
    batches = sorted((p for p in batches if re.fullmatch(r'batch_\d+', p.name)),
                     key=lambda p: int(p.name[6:]))
    for batch in batches:
        if not re.fullmatch(r'batch_\d+', batch.name) or not first <= int(batch.name[6:]) <= last:
            continue
        print('BATCH', batch, flush=True)
        for session in directories(batch):
            yield from depth_sources(session / 'depth')


def fingerprint(path):
    stat = path.stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def verify(output, ffmpeg, count, expected_mask, *, decode=True):
    report = json.loads((output / 'lerobot_depth.json').read_text(encoding='utf-8'))
    root = (output / report['session_root_relative']).resolve()
    for key in ('file', 'mask_file'):
        path = root / report['depth_storage'][key]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty output: {path}')
    if not decode:
        return
    print('VERIFY', output, f'({count} frames)', flush=True)
    digest = hashlib.sha256()
    frames = 0
    last = time.monotonic()
    for frame in read_episode_depth(output, ffmpeg=ffmpeg):
        digest.update(np.isfinite(frame).tobytes())
        frames += 1
        if time.monotonic() - last >= 10:
            print(f'VERIFY {frames}/{count}', flush=True)
            last = time.monotonic()
    if frames != count or digest.hexdigest() != expected_mask:
        raise ValueError('Decoded frame count or exact validity mask differs')


def clean_legacy(source, session, report):
    """Never recursively delete a computed legacy directory."""
    if source.is_symlink():
        raise ValueError('Legacy cleanup cannot follow a symlink')
    source = source.resolve()
    if not source.is_relative_to((session / 'depth').resolve()) or source.name not in CAMERAS:
        raise ValueError('Unsafe legacy cleanup path')
    names = {report['migration']['source_video'], source.name + '.parquet', 'lerobot_depth.json',
             'depth_hevc12.ffmpeg.log', Path(report['migration']['source_video']).stem + '.ffmpeg.log',
             Path(report['migration']['source_video']).stem + '_meta.json',
             'fast_foundation_stereo_video_meta.json'}
    files = list(source.iterdir())
    if any(p.is_symlink() or not p.is_file() or p.name not in names for p in files):
        raise ValueError('Unexpected legacy content; outputs retained, cleanup refused')
    # Video first; a retry can finish deleting remaining recognized sidecars.
    for name in ([report['migration']['source_video']] +
                 sorted(names - {report['migration']['source_video'], 'lerobot_depth.json'}) + ['lerobot_depth.json']):
        (source / name).unlink(missing_ok=True)
    source.rmdir()
    for parent in (source.parent, session / 'depth'):
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def convert(source, *, execute=False, delete_source=False, ffmpeg='ffmpeg', threads=2, preset='slow', verify_video=True):
    source = Path(source).resolve()
    session = source.parents[2]
    feature = source.name
    camera = CAMERAS[feature]
    metadata_path = source / 'lerobot_depth.json'
    report = json.loads(metadata_path.read_text(encoding='utf-8'))
    storage = report['depth_storage']
    video = source / storage['file']
    if video.parent != source or not re.fullmatch(r'episode_\d+\.mkv', video.name):
        raise ValueError('Expected a normalized episode_XXXXXX.mkv')
    episode_name = video.stem
    episode = session / 'data' / source.parent.name / episode_name
    output = session / 'annotation' / source.parent.name / feature / episode_name
    output_video = session / 'videos' / source.parent.name / feature / (episode_name + '.mp4')
    count, size = storage['frames'], (storage['width'], storage['height'])
    timestamps = pq.read_table(source / report['timestamp_file'])
    if feature != FEATURE:
        # L3 stores full source rows; Lightwheel's old normalizer already selected them.
        import pyarrow as pa
        import pyarrow.compute as pc
        if 'episode_index' in timestamps.column_names:
            timestamps = timestamps.filter(pc.equal(timestamps['episode_index'], int(episode_name[8:])))
        indices = report['processed_frames']
        if any(type(i) is not int or i < 0 or i >= timestamps.num_rows for i in indices):
            raise ValueError('Invalid L3 processed_frames source indices')
        timestamps = timestamps.take(pa.array(indices, type=pa.int64()))
    if count <= 0 or timestamps.num_rows != count or len(report['processed_frames']) != count:
        raise ValueError('Normalized timestamps, processed_frames and video count must match')
    if not np.all(np.isfinite(timestamps['timestamp'].to_numpy())):
        raise ValueError('Nonfinite source timestamps')
    if np.any(np.diff(timestamps['timestamp'].to_numpy()) < 0):
        raise ValueError('Source timestamps are not monotonic')
    if not execute:
        print('PLAN', video, '->', output_video, f'({count} frames; CRF 0 {preset})', flush=True)
        return 'planned'
    session.mkdir(parents=True, exist_ok=True)
    published = output / 'lerobot_depth.json'
    if published.exists():
        saved = json.loads(published.read_text(encoding='utf-8'))
        migration = saved.get('migration', {})
        if migration.get('source_video') != video.name or migration.get('source_directory') != source.relative_to(session).as_posix():
            raise FileExistsError('Destination belongs to another job; refusing overwrite')
        if video.exists() and migration['source_fingerprint'] != fingerprint(video):
            raise ValueError('Legacy source changed since publication')
        verify(output, ffmpeg, count, migration['validity_sha256'], decode=verify_video)
        if not pq.read_table(session / saved['timestamp_file']).equals(timestamps, check_metadata=True):
            raise ValueError('Published timestamp table differs from source')
        if delete_source:
            clean_legacy(source, session, saved)
        return 'reused'
    if output_video.exists():
        raise FileExistsError('Destination video exists without migration metadata')
    if not storage.get('lossless', False):
        raise ValueError('Source must be confirmed lossless before deriving validity from zero')
    initial = fingerprint(video)
    print('ENCODE', video, f'({count} frames; {size[0]}x{size[1]}; preset={preset}; threads={threads})', flush=True)
    with tempfile.TemporaryDirectory(prefix='.reencode-', dir=session) as tmp:
        work = Path(tmp)
        digest = hashlib.sha256()
        writer = HEVC12Writer(work / 'depth_hevc12.mp4', size=size,
            fps=storage.get('fps', 30), max_depth_m=storage.get('max_depth_m', 4.),
            ffmpeg=ffmpeg, preset=preset, threads=threads)
        actual = 0
        started = last = time.monotonic()
        try:
            for frame in _raw_frames(video, size=size, ffmpeg=ffmpeg, pixel_format='gray12le', dtype='<u2'):
                valid = frame != 0
                digest.update(valid.tobytes())
                depth = frame.astype(np.float32) / 1000.
                depth[~valid] = np.nan
                writer.write(depth)
                actual += 1
                now = time.monotonic()
                if actual == 1 or now - last >= 10:
                    speed = actual / max(now - started, .001)
                    print(f'ENCODE {actual}/{count} {speed:.2f} fps; ETA {(count-actual)/speed/60:.1f} min', flush=True)
                    last = now
        finally:
            print('FINISH ENCODER', actual, flush=True)
            writer.close()
        if actual != count or initial != fingerprint(video):
            raise ValueError('Source frame count mismatch or source changed during encoding')
        report['migration'] = dict(source_video=video.name,
            source_directory=source.relative_to(session).as_posix(),
            source_fingerprint=initial, validity_sha256=digest.hexdigest())
        report['timestamp_index'] = 'video_frame'
        report['depth_storage'].update(file='depth_hevc12.mp4', mask_file='valid_mask_ffv1.mkv',
            container='mp4', codec='hevc', lossless=False, crf=0, preset=preset, requires_valid_mask=True,
            video_index_to_source_index='source_index = processed_frames[video_index]')
        # Verify temporary video and mask before publishing any destination files.
        temporary_report = dict(report, session_root_relative='.')
        temporary_report['depth_storage'] = dict(report['depth_storage'], path_base='session')
        (work / 'lerobot_depth.json').write_text(json.dumps(temporary_report), encoding='utf-8')
        verify(work, ffmpeg, count, digest.hexdigest(), decode=verify_video)
        pq.write_table(timestamps, work / 'source_timestamps.parquet')
        (work / 'lerobot_depth.json').write_text(json.dumps(report), encoding='utf-8')
        for name in ('fast_foundation_stereo_video_meta.json', episode_name + '_meta.json'):
            if (source / name).exists():
                backend_meta = json.loads((source / name).read_text(encoding='utf-8'))
                backend_meta['depth_encoding'] = 'hevc12_crf0'
                (work / name).write_text(json.dumps(backend_meta, indent=2), encoding='utf-8')
        publish_video(work, output, session, episode, camera)
    verify(output, ffmpeg, count, digest.hexdigest(), decode=verify_video)
    saved = json.loads((output / 'lerobot_depth.json').read_text(encoding='utf-8'))
    if not pq.read_table(session / saved['timestamp_file']).equals(timestamps, check_metadata=True):
        raise ValueError('Published timestamp table differs from source')
    if delete_source:
        clean_legacy(source, session, report)
    print('DONE', output_video, flush=True)
    return 'converted'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--delete-source', action='store_true')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--encode-threads', type=int, default=2)
    parser.add_argument('--skip-video-verification', action='store_true',
                        help='Skip output video/mask decoding; retain encoder, file and timestamp checks')
    parser.add_argument('--preset', choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'], default='slow')
    parser.add_argument('--first-batch', type=int, default=1)
    parser.add_argument('--last-batch', type=int, default=9999999)
    args = parser.parse_args(argv)
    print(f'START root={args.root} workers={args.workers} preset={args.preset} execute={args.execute}', flush=True)
    if not args.root.is_dir() or min(args.workers, args.encode_threads) < 1 or not 1 <= args.first_batch <= args.last_batch:
        parser.error('Invalid root, worker count, thread count or batch range')
    if args.delete_source and not args.execute:
        parser.error('--delete-source requires --execute')
    def run(source):
        try:
            return convert(source, execute=args.execute, delete_source=args.delete_source,
                ffmpeg=args.ffmpeg, threads=args.encode_threads, preset=args.preset,
                verify_video=not args.skip_video_verification)
        except Exception as exc:
            print('FAILED', source, str(exc), file=sys.stderr, flush=True)
            return 'failed'
    counts = {}
    sources = iter(discover_sources(args.root, args.first_batch, args.last_batch))
    pending, exhausted, submitted = set(), False, 0
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < args.workers:
                source = next(sources, None)
                if source is None:
                    exhausted = True
                    break
                submitted += 1
                print('QUEUE', submitted, source, flush=True)
                pending.add(pool.submit(run, source))
            if pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    counts[result] = counts.get(result, 0) + 1
                print(f'STATUS submitted={submitted} active={len(pending)} completed={sum(counts.values())} results={counts}', flush=True)
    except KeyboardInterrupt:
        print('CANCEL: stopping FFmpeg children; legacy source files are retained.', file=sys.stderr, flush=True)
        cancel_active_processes()
        pool.shutdown(wait=True, cancel_futures=True)
        return 130
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    print(json.dumps(counts), flush=True)
    return int(not counts or counts.get('failed', 0) > 0)


if __name__ == '__main__':
    sys.exit(main())
