"""Remove specified Lightwheel release sidecars; preview unless --execute is given.

Only inspect root/session and root/batch_*/session, never recursively scan data
or videos. Stop annotation/enrichment writers before deleting their sidecars.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil


def directories(path):
    with os.scandir(path) as entries:
        return sorted(Path(e.path) for e in entries if e.is_dir(follow_symlinks=False))


def sessions(root, *, verbose=False):
    if (root / 'annotation').is_dir() or (root / 'meta').is_dir():
        yield root
        return
    for child in directories(root):
        if re.fullmatch(r'batch_\d+', child.name):
            if verbose:
                print('BATCH', child, flush=True)
            for session in directories(child):
                if (session / 'annotation').is_dir() or (session / 'meta').is_dir():
                    yield session
        elif (child / 'annotation').is_dir() or (child / 'meta').is_dir():
            yield child


def targets(session):
    annotation = session / 'annotation'
    if annotation.is_symlink() or (session / 'meta').is_symlink():
        raise ValueError(f'Refusing symlink annotation/meta directory: {session}')
    if annotation.is_dir():
        for path in sorted(annotation.glob('episode_*_structured.json.before_enrichment.bak')):
            if re.fullmatch(r'episode_\d+_structured\.json\.before_enrichment\.bak', path.name):
                yield path, False
    yield annotation / 'semantic_segments.json', False
    yield annotation / 'stereo_quality', True
    yield session / 'meta/guanglun_source.json', False
    yield session / 'meta/guanglun_video_validation.json', False


def clean_session(session, root, *, execute=False, verbose=False):
    counts = dict(files=0, directories=0, missing=0)
    for path, directory in targets(session):
        if path.is_symlink():
            raise ValueError(f'Refusing symlink target: {path}')
        if not path.exists():
            counts['missing'] += 1
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.is_relative_to(session.resolve()):
            raise ValueError(f'Target escapes dataset: {path}')
        if directory != path.is_dir() or (not directory and not path.is_file()):
            raise ValueError(f'Unexpected target type: {path}')
        if verbose:
            print('DELETE' if execute else 'PLAN', path, flush=True)
        if execute:
            if directory:
                shutil.rmtree(path)
            else:
                path.unlink()
        counts['directories' if directory else 'files'] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Delete listed targets; otherwise preview only')
    parser.add_argument('--verbose', action='store_true', help='Print individual paths and intermediate progress')
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error('root must be an existing directory')
    root = args.root.resolve()
    print(f'START root={root} execute={args.execute}', flush=True)
    totals = dict(sessions=0, files=0, directories=0, missing=0, failed=0)
    for session in sessions(root, verbose=args.verbose):
        totals['sessions'] += 1
        try:
            for key, value in clean_session(session, root, execute=args.execute, verbose=args.verbose).items():
                totals[key] += value
        except Exception as exc:
            totals['failed'] += 1
            print('FAILED', session, str(exc), flush=True)
        if args.verbose and totals['sessions'] % 100 == 0:
            print('STATUS', json.dumps(totals), flush=True)
    print('SUMMARY', json.dumps(totals), flush=True)
    return int(totals['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
