"""Run L3 headband stereo rectification using meta/calibrations.jsonl."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import hashlib
import shutil
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roboego_hand_vis.lerobot_headband_integrity.L1.run import discover
from roboego_hand_vis.lerobot_stereo_rectification.L3.compute import process_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--balance", type=float, default=0.)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--center-crop-calibration", action=argparse.BooleanOptionalAction,
                        default=True, help="Automatically center-crop to calibration resolution (default: enabled)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                        help="Validate and reuse complete rectification (default); --no-skip-existing forces recompute")
    parser.add_argument("--sessions-file", type=Path, help="JSON list of selected session paths")
    parser.add_argument("--work-root", type=Path, help="Work files and encoder logs outside the dataset")
    args = parser.parse_args(argv)
    if (args.workers < 1 or not math.isfinite(args.balance) or not 0 <= args.balance <= 1):
        parser.error("Invalid worker count or balance")
    sessions = discover(args.source, args.sessions_file)
    if not sessions:
        parser.error("No datasets with meta/info.json found")
    options = dict(balance=args.balance, ffmpeg=args.ffmpeg,
                   center_crop_calibration=args.center_crop_calibration, dry_run=args.dry_run,
                   skip_existing=args.skip_existing)
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work_root = args.work_root or Path(os.environ.get('L3_RECTIFY_WORK_ROOT') or Path.home()/'.cache/roboego/L3/rectify')
        futures = {pool.submit(process_session, s, dict(options, work_root=work_root/hashlib.sha256(str(s).encode()).hexdigest())): s for s in sessions}
        for future in as_completed(futures):
            session = futures[future]
            try:
                results = future.result()
                if not args.dry_run:
                    work = work_root/hashlib.sha256(str(session).encode()).hexdigest()
                    if work.exists():
                        shutil.rmtree(work)
                print(json.dumps(dict(session=str(session), status="dry_run" if args.dry_run else "done", results=results)), flush=True)
            except Exception as exc:
                failed += 1
                print(json.dumps(dict(session=str(session), status="failed", error=str(exc))), flush=True)
    print(json.dumps(dict(total=len(sessions), failed=failed)))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())

'''
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_stereo_rectification/L3/run.py \
  --source /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001 \
  --ffmpeg /share/project/liuyou/humancoin/envs/ego/bin/ffmpeg \
  --workers 4

'''
