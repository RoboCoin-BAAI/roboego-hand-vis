"""Run ROS-free VIO for one L1 LeRobot dataset or a batch directory."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roboego_hand_vis.lerobot_headband_integrity.L1.run import discover
from roboego_hand_vis.lerobot_vio.L1.compute import plan_session, run_session
from roboego_hand_vis.lerobot_vio.L1.quality import DEFAULTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--openvins-root', type=Path, default=Path(__file__).resolve().parents[4] / 'open_vins')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--input-mode', choices=('rectified', 'raw'), default='rectified')
    parser.add_argument('--raw-center-crop-calibration', action='store_true',
                        help='Confirm calibration describes the centered ROI of larger raw videos; lossless crop without undistortion')
    parser.add_argument('--start-s', type=float, default=0.)
    parser.add_argument('--duration-s', type=float)
    parser.add_argument('--stereo-tolerance-s', type=float, default=.001)
    parser.add_argument('--downsample-cameras', action='store_true')
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument('--zupt', action='store_true', help='Enable ZUPT for every run')
    policy.add_argument('--zupt-fallback', action='store_true', help='Run normal VIO first; retry with ZUPT only after quality or backend failure')
    parser.add_argument('--time-offset-mode', choices=('estimate', 'config'), default='estimate')
    parser.add_argument('--init-mode', choices=('static-zupt', 'dynamic', 'static-then-dynamic', 'dynamic-then-static', 'auto', 'static'), default=None,
                        help='Choose static+ZUPT, dynamic, or ordered fallback; default: static-then-dynamic')
    parser.add_argument('--timing-bootstrap', choices=('auto', 'off'), default='auto')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    for key, default in DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), type=float, default=default)
    parser.add_argument("--sessions-file", type=Path, help="JSON list of selected session paths")
    args = parser.parse_args(argv)
    if args.init_mode is None:
        args.init_mode = 'auto' if args.zupt or args.zupt_fallback else 'static-then-dynamic'
    if args.init_mode in ('static-zupt', 'dynamic', 'static-then-dynamic', 'dynamic-then-static') and (args.zupt or args.zupt_fallback):
        parser.error('Use --init-mode alone; do not combine with legacy --zupt/--zupt-fallback')
    if args.workers < 1 or any(not math.isfinite(v) or v < 0 for v in (args.start_s, args.stereo_tolerance_s)):
        parser.error('workers must be positive; start/tolerance must be finite and nonnegative')
    if args.duration_s is not None and (not math.isfinite(args.duration_s) or args.duration_s <= 0):
        parser.error('duration must be finite and positive')
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) < 0 for k in DEFAULTS) or args.min_valid_ratio > 1:
        parser.error('Invalid quality thresholds')
    sessions = discover(args.source, args.sessions_file)
    if not sessions:
        parser.error('No datasets with meta/info.json found')
    jobs, failed = [], 0
    for session in sessions:
        try:
            jobs.extend(plan_session(session, input_mode=args.input_mode, zupt=args.zupt))
        except Exception as exc:
            failed += 1
            print(json.dumps(dict(session=str(session), status='failed', error=str(exc))), flush=True)
    if args.dry_run:
        for plan in jobs:
            print(json.dumps(plan, default=str), flush=True)
        return int(bool(failed))
    options = {k: v for k, v in vars(args).items() if k not in ('source', 'workers', 'dry_run', 'input_mode', 'sessions_file') and k not in DEFAULTS}
    options['quality_thresholds'] = {k: getattr(args, k) for k in DEFAULTS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        groups = {}
        for plan in jobs:
            groups.setdefault(plan['session'], []).append(plan)
        futures = {pool.submit(run_session, plans, options): session for session, plans in groups.items()}
        for future in as_completed(futures):
            try:
                items = future.result()
            except Exception as exc:
                items = [dict(session=str(futures[future]), status='failed', error=str(exc))]
            for item in items:
                failed += item['status'] in ('failed', 'failed_quality')
                print(json.dumps(item), flush=True)
    print(json.dumps(dict(total_episodes=len(jobs), failed=failed)))
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())


'''
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_vio/L1/run.py \
  --source /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001 \
  --openvins-root /share/project/liuyou/humancoin/code/open_vins \
  --workers 8 \
  --input-mode rectified \
  --stereo-tolerance-s 0.001 \
  --time-offset-mode estimate \
  --timing-bootstrap auto \
  --zupt


/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_vio/L1/run.py \
  --source /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001/cp_0001_session_20261009_172052 \
  --openvins-root /share/project/liuyou/humancoin/code/open_vins \
  --workers 1 \
  --stereo-tolerance-s 0.001 \
  --time-offset-mode estimate \
  --timing-bootstrap auto

'''
