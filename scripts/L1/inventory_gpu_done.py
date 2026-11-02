"""List existing L1 depth/hand outputs by file presence only; never run inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import subprocess
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def save(path, values):
    temp = path.with_name('.' + path.name + '.' + uuid4().hex)
    temp.write_text(json.dumps(values, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def inspect_session(session, ffprobe=None):
    episodes = sorted(p for p in (session/'data').glob('chunk-*/episode_*') if p.is_dir())
    def present(stage, filenames):
        return bool(episodes) and all(
            all((session/stage/e.parent.name/e.name/name).is_file() for name in filenames)
            for e in episodes)
    return dict(session=str(session),
                depth_done=present('depth', ['lerobot_depth.json']),
                hand_done=present('hand', ['lerobot_hand.json', 'prediction.npz']),
                errors={})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('/share/project/liuyou/humancoin/data/L1_existing_paths.json'))
    parser.add_argument('--root', type=Path, default=Path('/share/project/liuyou/humancoin/data/L1_lerobot'))
    parser.add_argument('--ffprobe', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    rows = json.loads(args.manifest.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        parser.error('Expected a nonempty inventory array')
    root = args.root.resolve()
    sessions = [Path(row['session']).resolve() for row in rows]
    if len(sessions) != len(set(sessions)):
        parser.error('Duplicate session paths')
    batches = {}
    for session in sessions:
        if (not session.is_relative_to(root) or len(session.relative_to(root).parts) != 2
                or not session.parent.name.startswith('batch_') or not session.parent.name[6:].isdigit()
                or not session.name.startswith('cp_') or not session.is_dir()):
            parser.error('Invalid/missing session: ' + str(session))
        batches.setdefault(session.parent, []).append(session)
    total = dict(sessions=0, depth_done=0, hand_done=0)
    for batch, selected in sorted(batches.items()):
        results = []
        print(f'Checking {batch.name}: {len(selected)} sessions', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(lambda s: inspect_session(s, args.ffprobe), selected):
                results.append(result)
                if len(results) % 10 == 0:
                    print(f'{batch.name}: {len(results)}/{len(selected)} checked', flush=True)
        for stage in ('depth','hand'):
            done = sorted(r['session'] for r in results if r[stage+'_done'])
            save(batch/(stage+'_done.json'), done)
            total[stage+'_done'] += len(done)
        save(batch/'gpu_existing_validation.json', results)
        total['sessions'] += len(results)
        print(json.dumps(dict(batch=batch.name,sessions=len(results),
            depth_done=sum(r['depth_done'] for r in results),hand_done=sum(r['hand_done'] for r in results))),flush=True)
    print(json.dumps(total),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
