"""Check one L3 session or recursively check a batch directory."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roboego_hand_vis.lerobot_headband_integrity.L1.run import discover
from roboego_hand_vis.lerobot_headband_integrity.L3.compute import export_report


def worker(session, options):
    try:
        return export_report(session, **options)
    except Exception as exc:
        return dict(session=str(session), passed=False, error=f'{type(exc).__name__}: {exc}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--audio-duration-tolerance-s', type=float, default=5.0)
    parser.add_argument('--sessions-file', type=Path)
    parser.add_argument('--calibration-root', type=Path,
                        help='Donor search root across batches; defaults to the source batch parent')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1 or any(not math.isfinite(v) or v < 0 for v in
                              (args.audio_duration_tolerance_s,)):
        parser.error('workers must be positive; tolerances must be finite and nonnegative')
    try:
        sessions = discover(args.source, args.sessions_file)
        if not sessions:
            raise ValueError('No datasets with meta/info.json found')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for session in sessions:
            print(session)
        return 0
    options = dict(ffprobe=args.ffprobe,
                   duration_tolerance_s=args.audio_duration_tolerance_s)
    source = args.source.resolve()
    if (source / 'meta/info.json').is_file():
        default_root = source.parent.parent if source.parent.name.startswith('batch_') else source.parent
    else:
        default_root = source.parent if source.name.startswith('batch_') else source
    calibration_root = (args.calibration_root or default_root).resolve()
    # Snapshot original donors before workers can create repaired calibration files.
    needs_donors = any(not (session / 'meta/calibrations.jsonl').exists() for session in sessions)
    candidates = (set(calibration_root.glob('batch_*/*/meta/calibrations.jsonl')) |
                  set(calibration_root.glob('*/meta/calibrations.jsonl'))) if needs_donors else set()
    options['calibration_donors'] = [p.parent.parent for p in sorted(candidates)
                                    if not (p.parent.parent / 'annotation/l3_calibration_repair.json').exists()]
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, s, options): s for s in sessions}
        for future in as_completed(futures):
            try:
                item = future.result()
            except Exception as exc:
                item = dict(session=str(futures[future]), passed=False, error=str(exc))
            failed += not item['passed']
            print(json.dumps(item, ensure_ascii=False), flush=True)
    print(json.dumps(dict(total=len(sessions), passed=len(sessions)-failed, failed=failed)))
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())

'''
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_headband_integrity/L3/run.py \
  --source /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001 \
  --calibration-root /share/project/liuyou/humancoin/data/L3_lerobot \
  --workers 4
'''