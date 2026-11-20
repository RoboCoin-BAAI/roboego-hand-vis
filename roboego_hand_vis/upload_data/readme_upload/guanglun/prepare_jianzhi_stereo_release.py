"""Retain rectified stereo/depth, with external backups. No video decoding."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

KEEP = ('observation.images.head_left_rectified',
        'observation.images.head_right_rectified',
        'observation.images.depth_head_left_rectified')


def atomic(path, content):
    fd, name = tempfile.mkstemp(prefix='.stereo-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def plan(sample):
    info_path = sample / 'meta/info.json'
    original = info_path.read_bytes()
    info = json.loads(original)
    if info['codebase_version'] not in ('v3.1', '3.1', 'v2.1', '2.1'):
        raise ValueError('Unexpected codebase_version')
    features = info['features']
    removed = {k for k, v in features.items()
               if (v.get('dtype') == 'video' or k.startswith('observation.images.')) and k not in KEEP}
    moves = []
    for episode in (sample / 'data').glob('chunk-*/episode_*'):
        moves.extend(path for path in episode.glob('observation.images.*.parquet')
                     if path.stem not in KEEP)
    for chunk in (sample / 'videos').glob('chunk-*'):
        for path in chunk.iterdir():
            if path.name not in KEEP:
                if not path.is_dir():
                    raise ValueError(f'Unexpected video entry: {path}')
                moves.append(path)
    info = copy.deepcopy(info)
    info['features'] = {k: v for k, v in features.items() if k not in removed}
    info['codebase_version'] = 'v2.1' if info['codebase_version'].startswith('v') else '2.1'
    info['master_feature'] = KEEP[0]
    # User confirms exactly three video streams per episode; no media probing.
    info['total_videos'] = len(KEEP) * info['total_episodes']
    content = encoded(info)
    writes = {} if content.encode('utf-8') == original else {info_path: content}
    return writes, moves


def process(sample, root, backup, apply):
    relative = sample.relative_to(root)
    try:
        writes, moves = plan(sample)
        if apply and (writes or moves):
            target = backup / relative
            # Never overwrite an earlier backup on a repeated invocation.
            if target.exists():
                raise ValueError(f'Backup already exists: {target}')
            resolved_sample = sample.resolve()
            if not resolved_sample.is_relative_to(root):
                raise ValueError(f'Sample escapes publication root: {sample}')
            resolved_parents = {}
            for path in list(writes) + moves:
                if path.parent not in resolved_parents:
                    resolved_parents[path.parent] = path.parent.resolve()
                if path.is_symlink() or not resolved_parents[path.parent].is_relative_to(resolved_sample):
                    raise ValueError(f'Unsafe source: {path}')
            target.mkdir(parents=True)
            originals = {}
            moved = []
            try:
                for path in writes:
                    dest = target / path.relative_to(sample)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                    originals[path] = dest
                for path in moves:
                    dest = target / path.relative_to(sample)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if path.is_dir():
                        os.rename(path, dest)
                    else:
                        # Small sidecars: copy completely before unlinking. Avoid
                        # the shared filesystem's global cross-directory rename lock.
                        shutil.copy2(path, dest)
                        path.unlink()
                    moved.append((path, dest))
                for path, content in writes.items():
                    atomic(path, content)
                # Verify only the requested migration, without probing retained media.
                pending_writes, pending_moves = plan(sample)
                if pending_writes or pending_moves:
                    raise ValueError('Post-write verification failed')
            except Exception:
                for path, dest in originals.items():
                    shutil.copy2(dest, path)
                for path, dest in reversed(moved):
                    if dest.is_dir():
                        os.rename(dest, path)
                    else:
                        shutil.copy2(dest, path)
                        dest.unlink()
                raise
        return dict(path=str(relative), status='ok', metadata=len(writes), moved=len(moves))
    except Exception as exc:
        return dict(path=str(relative), status='error', error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=71)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root, backup = args.root.resolve(), args.backup_root.resolve()
    if backup == root or backup.is_relative_to(root):
        parser.error('Backup must be outside publication root')
    samples = []
    for number in range(args.start, args.end + 1):
        batch = root / f'batch_{number:07d}'
        if not batch.is_dir():
            parser.error(f'Missing batch: {batch}')
        found = sorted(p for p in batch.iterdir() if p.is_dir() and p.name != '_batch_logs')
        if not found:
            parser.error(f'Empty batch: {batch}')
        samples.extend(found)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    errors = changed = moved = 0
    with args.report.open('w', encoding='utf-8') as report, ThreadPoolExecutor(args.workers) as pool:
        for number, result in enumerate(pool.map(lambda p: process(p, root, backup, args.apply), samples), 1):
            report.write(json.dumps(result, ensure_ascii=False) + '\n')
            report.flush()
            errors += result['status'] == 'error'
            changed += result.get('metadata', 0) > 0 or result.get('moved', 0) > 0
            moved += result.get('moved', 0)
            if number % 100 == 0 or result['status'] == 'error':
                print(json.dumps(dict(completed=number, total=len(samples), errors=errors,
                                      last=result)), flush=True)
    print(json.dumps(dict(total=len(samples), errors=errors, changed=changed,
                          moved=moved, apply=args.apply)), flush=True)
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
