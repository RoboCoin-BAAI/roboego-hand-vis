"""Single-session or batch GPU inference for L3 LeRobot rectified videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roboego_hand_vis.lerobot_headband_integrity.L1.run import discover
from roboego_hand_vis.lerobot_fast_foundation_depth.L3.compute import DEFAULT_CONFIG, plan_session, run_session, CAMERAS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model-root", type=Path)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--gpus", help="Physical GPU IDs, e.g. 0,1; one inference subprocess per listed GPU")
    p.add_argument("--num-frames", default="auto")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--right-frame-offset", type=int, default=0,
                   help="Right index = left index + offset; -1 pairs left[i] with right[i-1]")
    p.add_argument("--engine", choices=("throughput", "legacy"), default="throughput")
    p.add_argument("--prefetch", type=int, default=4, help="Decoded stereo frames queued per GPU")
    p.add_argument("--write-queue", type=int, default=4, help="Pending output frames per GPU")
    p.add_argument("--cpu-threads", type=int, default=2, help="CPU and decode threads per GPU worker")
    p.add_argument("--depth-format", choices=("files", "hevc12"), default="files")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--hevc12-preset", default="ultrafast")
    p.add_argument("--encode-threads", type=int, default=2)
    p.add_argument("--save-vis", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable preview video explicitly (default: disabled); overrides YAML")
    p.add_argument("--preview-every-n-frames", type=int, default=None)
    p.add_argument("--vis-min-depth", type=float, default=.2, help="Preview near limit in meters")
    p.add_argument("--vis-max-depth", type=float, default=1., help="Preview far limit in meters")
    p.add_argument("--vis-mode", choices=("inverse", "linear"), default="linear")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--sessions-file", type=Path, help="JSON list of selected session paths")
    args = p.parse_args(argv)
    if args.start < 0 or (args.num_frames != "auto" and (not args.num_frames.isdigit() or int(args.num_frames) < 1)):
        p.error("start must be nonnegative; num-frames must be auto or a positive integer")
    if min(args.prefetch, args.write_queue, args.cpu_threads, args.encode_threads) < 1:
        p.error("prefetch, write-queue and cpu-threads must be positive")
    if args.preview_every_n_frames is not None and args.preview_every_n_frames < 1:
        p.error("preview-every-n-frames must be positive")
    if not all(math.isfinite(v) for v in (args.vis_min_depth, args.vis_max_depth)) or not 0 <= args.vis_min_depth < args.vis_max_depth:
        p.error("Preview range must satisfy 0 <= vis-min-depth < vis-max-depth")
    if args.vis_mode == "inverse" and args.vis_min_depth <= 0:
        p.error("Inverse preview requires vis-min-depth > 0")
    if args.engine == "legacy" and args.depth_format == "hevc12":
        p.error("Direct HEVC12 output requires --engine throughput")
    if args.engine == "legacy" and args.right_frame_offset:
        p.error("Frame offset trials require --engine throughput")
    devices = [d.strip() for d in args.gpus.split(",")] if args.gpus else [None]
    if args.gpus and (any(not d.strip() for d in devices) or len(set(devices)) != len(devices)):
        p.error("GPU IDs must be nonempty and unique")
    sessions = discover(args.source, args.sessions_file)
    if not sessions:
        p.error("No datasets with meta/info.json found")
    if args.worker:
        if len(sessions) != 1:
            p.error("Worker expects exactly one dataset")
        print(json.dumps(run_session(sessions[0], config=args.config, model_root=args.model_root,
              ckpt=args.ckpt, start=args.start, num_frames=args.num_frames,
              save_vis=args.save_vis, preview_every_n_frames=args.preview_every_n_frames)), flush=True)
        return 0
    if args.dry_run:
        failed = 0
        for session in sessions:
            try:
                print(json.dumps(dict(session=str(session), episodes=plan_session(session))), flush=True)
            except Exception as exc:
                failed += 1
                print(json.dumps(dict(session=str(session), error=str(exc))), flush=True)
        return int(bool(failed))

    if args.engine == "throughput":
        from roboego_hand_vis.lerobot_fast_foundation_depth.L1.throughput import run_jobs
        jobs, invalid = [], 0
        for session in sessions:
            try:
                for episode in plan_session(session):
                    jobs.append(dict(task_id=len(jobs), session=str(session), episode=episode["episode"]))
            except Exception as exc:
                invalid += 1
                print(json.dumps(dict(session=str(session), status="failed", error=str(exc))), flush=True)
        options = dict(camera_names=CAMERAS, config=str(args.config.resolve()), model_root=str(args.model_root.resolve()) if args.model_root else None,
            ckpt=str(args.ckpt.resolve()) if args.ckpt else None, num_frames=args.num_frames, start=args.start,
            prefetch=args.prefetch, write_queue=args.write_queue, cpu_threads=args.cpu_threads,
            depth_format=args.depth_format, ffmpeg=args.ffmpeg, hevc12_preset=args.hevc12_preset,
            encode_threads=args.encode_threads, save_vis=args.save_vis,
            preview_every_n_frames=args.preview_every_n_frames,
            vis_min_depth=args.vis_min_depth, vis_max_depth=args.vis_max_depth, vis_mode=args.vis_mode,
            right_frame_offset=args.right_frame_offset)
        reports = run_jobs(jobs, devices, options)
        failed = invalid + sum(not r["passed"] for r in reports)
        print(json.dumps(dict(total_episodes=len(jobs), failed=failed)), flush=True)
        return int(bool(failed))

    def lane(index):
        failed = 0
        for session in sessions[index::len(devices)]:
            log = session / "annotation/lerobot_fast_foundation_depth.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--source", str(session),
                       "--config", str(args.config.resolve()), "--start", str(args.start), "--num-frames", args.num_frames]
            for flag, value in (("--model-root", args.model_root), ("--ckpt", args.ckpt)):
                if value is not None:
                    command.extend([flag, str(value.resolve())])
            if args.save_vis is not None:
                command.append("--save-vis" if args.save_vis else "--no-save-vis")
            if args.preview_every_n_frames is not None:
                command.extend(["--preview-every-n-frames", str(args.preview_every_n_frames)])
            env = os.environ.copy()
            if devices[index] is not None:
                env["CUDA_VISIBLE_DEVICES"] = devices[index].strip()
            print(f"[start] {session.name}, GPU={env.get('CUDA_VISIBLE_DEVICES', 'inherited')}", flush=True)
            try:
                with log.open("w", encoding="utf-8") as f:
                    code = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
            except OSError as exc:
                code = -1
                log.write_text(str(exc), encoding="utf-8")
            failed += bool(code)
            print(json.dumps(dict(session=str(session), status="failed" if code else "done", log=str(log))), flush=True)
        return failed

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        failed = sum(pool.map(lane, range(len(devices))))
    print(json.dumps(dict(total=len(sessions), failed=failed)))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())


# /share/project/liuyou/humancoin/envs/ego/bin/python \
#   /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_fast_foundation_depth/L3/run.py \
#   --source /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001 \
#   --model-root /share/project/liuyou/humancoin/code/Fast-FoundationStereo \
#   --ckpt /share/project/liuyou/humancoin/code/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
#   --engine throughput --gpus 0,1,2,3,4,5,6,7 \
#   --num-frames auto --depth-format hevc12 \
#   --ffmpeg /share/project/liuyou/humancoin/envs/ego/bin/ffmpeg
# Add --dry-run to validate input paths without loading the model.
