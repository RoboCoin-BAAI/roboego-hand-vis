#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.pre_semantic_gate import PreSemanticGateConfig, run_pre_semantic_quality_gate


def default_output_json_for_video(video_path: Path) -> Path:
    video_path = Path(video_path)
    if video_path.parent.name == "videos":
        return video_path.parent.parent / "annotation" / f"video_simple_quality_{video_path.stem}.json"
    return video_path.parent / f"video_simple_quality_{video_path.stem}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video simple quality gate on a single video.")
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--output_json",
        default="",
        help="Output report path. Defaults to <session>/annotation/video_simple_quality_<camera>.json for session videos.",
    )
    parser.add_argument("--sample_every_s", type=float, default=1.0)
    parser.add_argument("--dark_threshold", type=int, default=10)
    parser.add_argument("--bright_threshold", type=int, default=245)
    parser.add_argument("--max_black_ratio", type=float, default=0.98)
    parser.add_argument("--max_overexposed_ratio", type=float, default=0.98)
    parser.add_argument("--min_texture_std", type=float, default=3.0)
    parser.add_argument("--freeze_absdiff_threshold", type=float, default=1.5)
    parser.add_argument("--low_motion_absdiff_threshold", type=float, default=4.0)
    parser.add_argument("--min_freeze_duration_s", type=float, default=5.0)
    parser.add_argument("--min_bad_visual_duration_s", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_json = Path(args.output_json) if args.output_json else default_output_json_for_video(video_path)
    config = PreSemanticGateConfig(
        sample_every_s=args.sample_every_s,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        max_black_ratio=args.max_black_ratio,
        max_overexposed_ratio=args.max_overexposed_ratio,
        min_texture_std=args.min_texture_std,
        freeze_absdiff_threshold=args.freeze_absdiff_threshold,
        low_motion_absdiff_threshold=args.low_motion_absdiff_threshold,
        min_freeze_duration_s=args.min_freeze_duration_s,
        min_bad_visual_duration_s=args.min_bad_visual_duration_s,
    )
    report = run_pre_semantic_quality_gate(
        video_path=video_path,
        output_json=output_json,
        config=config,
    )
    summary = {
        "output_json": str(output_json),
        "decode_ok": report["checks"]["decode_ok"],
        "sampled_frame_count": report["checks"]["sampled_frame_count"],
        "drop_interval_count": sum(1 for item in report["intervals"] if item["route_hint"] == "drop_before_qwen"),
        "aux_interval_count": sum(1 for item in report["intervals"] if item["route_hint"] == "pass_to_qwen_with_aux"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
