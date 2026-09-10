#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.joints21_pose_gate import (  # noqa: E402
    build_joints21_pose_quality,
    write_joints21_pose_quality_outputs,
)
from roboego_hand_vis.egohand.orientation import load_mano_regressors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate final HaMeR hands using lightweight joints21 pose checks.")
    parser.add_argument("--final_tracks_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--mano_model_dir", default="../hamer/_DATA/data/mano")
    parser.add_argument("--max_bone_length_ratio", type=float, default=8.0)
    parser.add_argument("--max_joint_local_deviation", type=float, default=7.0)
    parser.add_argument("--max_wrist_local_deviation_m", type=float, default=0.6)
    parser.add_argument("--max_palm_local_deviation_m", type=float, default=0.6)
    parser.add_argument("--max_fingertip_local_deviation_m", type=float, default=0.7)
    parser.add_argument("--min_finger_direction_dot", type=float, default=-0.15)
    parser.add_argument("--drop_fingertips_below_wrist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_height", type=float, default=0.0)
    parser.add_argument("--max_suspicious_wrist_y_ratio", type=float, default=0.55)
    parser.add_argument("--min_fingertip_below_wrist_ratio", type=float, default=1.0)
    parser.add_argument("--min_fingertip_below_wrist_delta", type=float, default=0.02)
    parser.add_argument("--local_window_radius", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_tracks_json = Path(args.final_tracks_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = json.loads(final_tracks_json.read_text(encoding="utf-8"))
    regressor_by_side = load_mano_regressors(Path(args.mano_model_dir).expanduser())
    result = build_joints21_pose_quality(
        payload,
        start=args.start,
        num_frames=args.num_frames,
        regressor_by_side=regressor_by_side,
        max_bone_length_ratio=args.max_bone_length_ratio,
        max_joint_local_deviation=args.max_joint_local_deviation,
        max_wrist_local_deviation_m=args.max_wrist_local_deviation_m,
        max_palm_local_deviation_m=args.max_palm_local_deviation_m,
        max_fingertip_local_deviation_m=args.max_fingertip_local_deviation_m,
        min_finger_direction_dot=args.min_finger_direction_dot,
        drop_fingertips_below_wrist=args.drop_fingertips_below_wrist,
        image_height=args.image_height,
        max_suspicious_wrist_y_ratio=args.max_suspicious_wrist_y_ratio,
        min_fingertip_below_wrist_ratio=args.min_fingertip_below_wrist_ratio,
        min_fingertip_below_wrist_delta=args.min_fingertip_below_wrist_delta,
        local_window_radius=args.local_window_radius,
    )
    result["source_final_tracks_json"] = str(final_tracks_json)
    outputs = write_joints21_pose_quality_outputs(result, output_dir)
    print(json.dumps({"summary": result["summary"], "outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
