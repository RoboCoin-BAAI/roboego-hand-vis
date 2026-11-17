"""Register depth in meta/info.json and generate L3 meta/readme.json.

python scripts/update_depth_meta.py --root /path/to/session --execute
Also accepts a dataset root containing multiple sessions. No video is rewritten.
"""
import argparse
import copy
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roboego_hand_vis.lerobot_fast_foundation_depth.base.metadata import video_feature_definition


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def session_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes session: {relative}')
    return path


def is_junction(path):
    return getattr(path, 'is_junction', lambda: False)()


def discover_sessions(root, first_batch=1, last_batch=9999999):
    """Inspect only dataset/batch/session levels, never descend into session data."""
    if (root / 'meta/info.json').is_file():
        yield root
        return
    children = sorted(p for p in root.iterdir()
                      if p.is_dir() and not p.name.startswith('.') and not p.is_symlink())
    for child in children:
        match = re.fullmatch(r'batch_(\d+)', child.name)
        if match:
            if first_batch <= int(match[1]) <= last_batch:
                yield from discover_sessions(child, first_batch, last_batch)
        elif (child / 'meta/info.json').is_file():
            yield child


def build_update(root):
    pending = next((root / 'videos').glob('chunk-*/*/episode_*.mp4.temp'), None)
    if pending is not None:
        raise ValueError(f'Video files are still incomplete (.temp): {pending}')
    info_path = root / 'meta' / 'info.json'
    original = read_json(info_path)
    updated = copy.deepcopy(original)
    features = updated['features']
    reports = sorted((root / 'annotation').glob('chunk-*/*/episode_*/lerobot_depth.json'))
    if not reports:
        raise ValueError(f'No published depth reports: {root}')
    definitions = {}
    for path in reports:
        report = read_json(path)
        storage = report['depth_storage']
        feature = report['depth_feature']
        if storage.get('path_base') != 'session':
            raise ValueError(f'Expected session-relative storage: {path}')
        for relative in (storage['file'], storage['mask_file'], report['timestamp_file']):
            target = session_file(root, relative)
            if not target.is_file() or target.stat().st_size == 0:
                raise ValueError(f'Missing or empty published file: {target}')
        chunk, episode = path.parents[2].name, path.parent.name
        expected = f'videos/{chunk}/{feature}/{episode}.mp4'
        if storage['file'] != expected:
            raise ValueError(f'Unexpected video layout: {path}')
        if len(report['processed_frames']) != storage['frames']:
            raise ValueError(f'Frame count mismatch: {path}')
        definition = video_feature_definition(storage)
        if feature in definitions and definitions[feature] != definition:
            raise ValueError(f'Inconsistent depth encoding across episodes: {feature}')
        definitions[feature] = definition
    for feature, definition in definitions.items():
        features.setdefault(feature, {}).update(definition)
        for key in ('depth_metadata_path', 'mask_path', 'timestamp_index', 'frame_mapping'):
            features[feature].pop(key, None)
    # total_videos counts episode video assets, not mask sidecars.
    known = {s['depth_feature'] for s in map(read_json, reports)}
    videos = set()
    for path in (root / 'videos').glob('chunk-*/*/episode_*.mp4'):
        if path.parent.name in features and features[path.parent.name].get('dtype') == 'video':
            videos.add(path.relative_to(root).as_posix())
    updated['total_videos'] = len(videos)
    return original, updated, sorted(known)


def cleanup_candidates(root):
    """Select only requested artifacts, without descending into dot directories."""
    candidates = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            if path.is_symlink() or is_junction(path):
                dirs.remove(name)
                continue
            if name.startswith('.'):
                candidates.add(path)
                dirs.remove(name)
        candidates.update(parent / name for name in names if name.startswith('.'))
    candidates.update(root.glob('calibrations/frame_mapping.parquet'))
    for name in ('hand_left_points_provenance.json', 'hand_right_points_provenance.json'):
        candidates.update(root.glob(f'data/chunk-*/episode_*/{name}'))
    for camera in ('headband_left_rgb_rectified', 'headband_right_rgb_rectified'):
        candidates.update(root.glob(f'videos/chunk-*/observation.images.{camera}/episode_*.ffmpeg.log'))
    return sorted(candidates)


def clean_session(root, execute=False):
    root = root.resolve()
    for path in cleanup_candidates(root):
        # Verify the final absolute target before any recursive deletion.
        resolved = path.resolve()
        if resolved == root or not resolved.is_relative_to(root):
            raise ValueError(f'Cleanup path escapes session: {path}')
        print(f'{"DELETE" if execute else "PLAN DELETE"} {path}', flush=True)
        if execute:
            if path.is_dir() and not path.is_symlink():
                # Refuse directory links anywhere inside a recursive target.
                for directory, dirs, _ in os.walk(path, followlinks=False):
                    if any((Path(directory) / name).is_symlink()
                           or is_junction(Path(directory) / name) for name in dirs):
                        raise ValueError(f'Cleanup directory contains links: {path}')
                shutil.rmtree(path)
            else:
                path.unlink()


def update_session(root, execute=False):
    original, updated, features = build_update(root)
    if original == updated:
        print(f'UNCHANGED {root}')
    else:
        print(f'{"UPDATE" if execute else "PLAN"} {root}: {", ".join(features)}; '
              f'total_videos {original.get("total_videos")} -> {updated["total_videos"]}')
    if not execute:
        clean_session(root, execute=False)
        print(f'PLAN generate L3 README: {root / "meta/readme.json"}')
        return
    if original != updated:
        write_atomic(root / 'meta' / 'info.json', updated)
    clean_session(root, execute=True)
    # Refresh the README even when info.json was already up to date.
    builder = Path(__file__).resolve().parents[1] / 'roboego_hand_vis/build_json/L3/build_readme.py'
    subprocess.run([sys.executable, str(builder), '--root', str(root)], check=True)


def write_atomic(target, value):
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix='.depth-meta-', suffix='.tmp', delete=False) as stream:
            name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(name, target)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--execute', action='store_true', help='Delete artifacts, update metadata and generate README')
    parser.add_argument('--first-batch', type=int, default=1)
    parser.add_argument('--last-batch', type=int, default=9999999)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir() or args.first_batch > args.last_batch:
        parser.error('Root must exist and first-batch must not exceed last-batch')
    sessions = list(discover_sessions(root, args.first_batch, args.last_batch))
    if not sessions:
        parser.error(f'No meta/info.json found under {root}')
    failed = done = 0
    print(f'SESSIONS {len(sessions)}; execute={args.execute}', flush=True)
    for index, session in enumerate(sessions, 1):
        print(f'SESSION {index}/{len(sessions)} {session}', flush=True)
        try:
            update_session(session, args.execute)
            done += 1
        except Exception as error:
            print(f'ERROR {session}: {error}', flush=True)
            failed += 1
    print(json.dumps(dict(total=len(sessions), succeeded=done, failed=failed,
                         execute=args.execute)), flush=True)
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
