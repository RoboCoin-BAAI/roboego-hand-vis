"""CPU multiprocess quality checks for one dataset or a batch tree."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roboego_hand_vis.lerobot_headband_integrity.L1.compute import export_report


def discover(source, sessions_file=None):
    source = Path(source).resolve()
    if sessions_file is not None:
        values = json.loads(Path(sessions_file).read_text(encoding="utf-8"))
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("sessions-file must be a JSON list of paths")
        sessions = sorted({Path(v).resolve() for v in values})
        for session in sessions:
            if not session.is_relative_to(source) or not (session / "meta/info.json").is_file():
                raise ValueError(f"Invalid selected session: {session}")
        return sessions
    if (source / "meta/info.json").is_file():
        return [source]
    return sorted({p.parent.parent for p in source.rglob("meta/info.json")
                   if not any(part.startswith(".") for part in p.relative_to(source).parts)})


def worker(session, options):
    try:
        return export_report(session, **options)
    except Exception as exc:
        return dict(session=str(session), passed=False, error=f"{type(exc).__name__}: {exc}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Single dataset or batch directory")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--timestamp-tolerance-s", type=float, default=0.02)
    parser.add_argument("--audio-duration-tolerance-s", type=float, default=5.0)
    parser.add_argument("--imu-gap-frames", type=float, default=10)
    parser.add_argument('--full-video-check', action='store_true', help='Force video decode counting instead of reusing file-bound receipts')
    parser.add_argument("--dry-run", action="store_true", help="List datasets without running checks or writing reports")
    parser.add_argument("--sessions-file", type=Path, help="JSON list of selected session paths")
    parser.add_argument('--repair-calibration-root', type=Path,
                        help='Opt in: repair incomplete IMU/camera-IMU calibration from same-cp QC-passed donors under this root')
    parser.add_argument('--calibration-donor-cache', type=Path, help='Shared immutable donor snapshot for this pipeline run')
    args = parser.parse_args(argv)
    options = dict(ffprobe=args.ffprobe, tolerance_s=args.timestamp_tolerance_s,
                   duration_tolerance_s=args.audio_duration_tolerance_s, imu_gap_frames=args.imu_gap_frames)
    if args.workers < 1 or any(not math.isfinite(v) or v < 0 for k, v in options.items() if k != "ffprobe"):
        parser.error("workers must be positive and tolerances must be finite and nonnegative")
    sessions = discover(args.source, args.sessions_file)
    options['full_video_check'] = args.full_video_check
    if not sessions:
        parser.error("No datasets with meta/info.json found")
    if args.dry_run:
        for session in sessions:
            print(session)
        return 0
    if args.repair_calibration_root:
        repair_root = args.repair_calibration_root.resolve()
        if any(not s.is_relative_to(repair_root) for s in sessions):
            parser.error('Selected sessions must be inside repair-calibration-root')
        options['validate_vio_calibration'] = True
    failed = warnings = 0
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, s, options): s for s in sessions}
        for future in as_completed(futures):
            try:
                item = future.result()
            except Exception as exc:
                item = dict(session=str(futures[future]), passed=False, error=str(exc))
            results[str(futures[future])] = item
            print('[qc-check] ' + str(futures[future]) + ': ' + ('passed' if item['passed'] else 'failed'), flush=True)
    if args.repair_calibration_root:
        from roboego_hand_vis.lerobot_headband_integrity.L1.calibration_repair import repair_sessions
        repairs = repair_sessions(sessions, repair_root, donor_cache=args.calibration_donor_cache)
        for session, repair in repairs.items():
            print('[calibration-repair] ' + session + ': ' + json.dumps(repair, ensure_ascii=False), flush=True)
            if repair.get('applied'):
                results[session] = worker(Path(session), options)
    for item in results.values():
        failed += not item['passed']
        warnings += bool(item.get('warnings'))
        print(json.dumps(item, ensure_ascii=False), flush=True)
    print(json.dumps(dict(total=len(sessions), passed=len(sessions) - failed,
                          failed=failed, sessions_with_warnings=warnings)))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())


'''
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_headband_integrity/L1/run.py \
  --source /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001 \
  --workers 4
'''
