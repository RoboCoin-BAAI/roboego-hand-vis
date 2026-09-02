#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SAM3_SCRIPT = PROJECT_ROOT / "scripts/object/run_sam3_video.py"

from roboego_hand_vis.objects.sam3_hand_clip import (
    build_sam3_hand_clip_plan,
    build_sam3_hand_clip_plans_from_stage1,
    combine_sam3_object_tracks,
    find_window_by_id,
    infer_window_id_from_dir,
    infer_windows_json_from_hand_window_dir,
    materialize_interval_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 object segmentation on a hand clip/window.")
    parser.add_argument("--hand_window_dir", required=True)
    parser.add_argument("--stage1_actions", required=True)
    parser.add_argument("--object_name", default="")
    parser.add_argument("--object_shape", default="")
    parser.add_argument("--object_color", default="")
    parser.add_argument("--auto_from_stage1", action="store_true")
    parser.add_argument("--sam3_script", default=str(DEFAULT_SAM3_SCRIPT))
    parser.add_argument("--windows_json", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--object_id", default="object_0")
    parser.add_argument("--python", default="python")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--prompt_frame_idx", type=int, default=None)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--no_offload_video_to_cpu", action="store_true")
    parser.add_argument("--clip_by_intervals", action="store_true")
    parser.add_argument("--interval_expand_s", type=float, default=0.0)
    parser.add_argument("--interval_merge_gap_s", type=float, default=0.0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hand_window_dir = Path(args.hand_window_dir)
    windows_json = Path(args.windows_json) if args.windows_json else infer_windows_json_from_hand_window_dir(hand_window_dir)
    window_id = infer_window_id_from_dir(hand_window_dir)
    window = find_window_by_id(windows_json, window_id)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.auto_from_stage1:
        plans = build_sam3_hand_clip_plans_from_stage1(
            hand_window_dir=hand_window_dir,
            stage1_actions=Path(args.stage1_actions),
            window=window,
            output_root=output_dir,
            sam3_script=Path(args.sam3_script),
            python_executable=args.python,
            fps=args.fps,
            prompt_frame_idx=args.prompt_frame_idx,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            no_offload_video_to_cpu=args.no_offload_video_to_cpu,
            clip_by_intervals=args.clip_by_intervals,
            interval_expand_s=args.interval_expand_s,
            interval_merge_gap_s=args.interval_merge_gap_s,
        )
    else:
        if not args.object_name:
            raise SystemExit("--object_name is required unless --auto_from_stage1 is set")
        plans = [
            build_sam3_hand_clip_plan(
                hand_window_dir=hand_window_dir,
                stage1_actions=Path(args.stage1_actions),
                window=window,
                object_name=args.object_name,
                object_shape=args.object_shape,
                object_color=args.object_color,
                output_dir=output_dir,
                sam3_script=Path(args.sam3_script),
                python_executable=args.python,
                object_id=args.object_id,
                fps=args.fps,
                prompt_frame_idx=args.prompt_frame_idx,
                checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                no_offload_video_to_cpu=args.no_offload_video_to_cpu,
            )
        ]

    result = {
        "mode": "auto_from_stage1" if args.auto_from_stage1 else "manual",
        "plan_count": len(plans),
        "plans": [
            {
                "object_id": plan.object_id,
                "prompt_text": plan.prompt_text,
                "prompt_frame_idx": plan.prompt_frame_idx,
                "output_dir": str(plan.output_dir),
                "metadata": str(plan.metadata_path),
                "command": plan.command,
            }
            for plan in plans
        ],
        "dry_run": args.dry_run,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.dry_run:
        for plan in plans:
            command = list(plan.command)
            if plan.object_interval is not None:
                with tempfile.TemporaryDirectory(prefix=f"sam3_{plan.object_id}_") as temp_dir:
                    clip_frames_dir = materialize_interval_frames(plan.frames_dir, Path(temp_dir), plan.object_interval)
                    command[command.index("--video") + 1] = str(clip_frames_dir)
                    subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
            else:
                subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
        if len(plans) > 1 and not args.clip_by_intervals:
            combined = combine_sam3_object_tracks(
                object_dirs=[plan.output_dir for plan in plans],
                frames_dir=plans[0].frames_dir,
                output_dir=plans[0].output_dir.parent / "combined",
            )
            print(json.dumps({"combined": combined}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
