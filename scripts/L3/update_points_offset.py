"""Recompute existing L3 points using the current mounting offset, without other pipeline stages."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roboego_hand_vis.lerobot_hand_pose.L3.convert_world_points import discover, worker
from roboego_hand_vis.lerobot_hand_pose.L3.world_points import T_TRACKER_WRIST, DEFAULT_MAX_GAP_S


def update_session(session, dry_run=False):
    try:
        episodes = sorted((session / 'data').glob('chunk-*/episode_*'))
        hands = [h for h in ('left', 'right') if any(
            (ep / f'observation.hand_{h}_points.parquet').exists() for ep in episodes)]
        if not hands:
            return dict(session=str(session), status='skipped_no_points')
        for ep in episodes:
            for hand in hands:
                if not (ep / f'observation.hand_{hand}_points.parquet').is_file():
                    raise ValueError(f'Incomplete existing points: {ep}')
        gap = DEFAULT_MAX_GAP_S
        side = 'both' if len(hands) == 2 else hands[0]
        if dry_run:
            return dict(session=str(session), status='would_update', side=side,
                        t_tracker_wrist_m=T_TRACKER_WRIST.tolist(), max_gap_s=gap)
        return worker(session, side, gap, True)
    except Exception as exc:
        return dict(session=str(session), status='failed', error=str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Session, batch, or dataset root')
    parser.add_argument('--batch-range', help='Inclusive batch range under source, e.g. 10:14')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        if args.batch_range:
            first, last = map(int, args.batch_range.split(':'))
            if first < 1 or last < first:
                raise ValueError('Invalid batch range')
            sessions = [s for n in range(first, last + 1)
                        for s in discover(args.source / f'batch_{n:07d}')]
        else:
            sessions = discover(args.source)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(update_session, sessions, [args.dry_run] * len(sessions)):
            print(json.dumps(result, ensure_ascii=False), flush=True)
            failed += result['status'] == 'failed'
    print(json.dumps(dict(total=len(sessions), failed=failed)))
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
