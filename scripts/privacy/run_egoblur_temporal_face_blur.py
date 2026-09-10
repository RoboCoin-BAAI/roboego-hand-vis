from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.privacy.egoblur_temporal import FaceBlurConfig, run_egoblur_temporal_face_blur


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EgoBlur face anonymization with sparse detection and temporal interpolation.",
    )
    parser.add_argument("--input_video", required=True, type=Path)
    parser.add_argument("--output_video", required=True, type=Path)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Temporal blur report. Defaults to <output_video_stem>_egoblur_temporal_report.json.",
    )
    parser.add_argument(
        "--egoblur_root",
        type=Path,
        default=Path("../EgoBlur"),
    )
    parser.add_argument(
        "--face_model_path",
        type=Path,
        default=Path("../EgoBlur/models/ego_blur_face_gen2.jit"),
    )
    parser.add_argument("--camera_name", default="camera-rgb")
    parser.add_argument("--detect_every_n_frames", type=int, default=3)
    parser.add_argument("--face_model_score_threshold", type=float, default=0.4)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--scale_factor_detections", type=float, default=1.35)
    parser.add_argument(
        "--max_gap_frames",
        type=int,
        default=12,
        help="Maximum missing-frame gap to fill by interpolation after sparse detection.",
    )
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json
    if output_json is None:
        output_json = args.output_video.with_name(f"{args.output_video.stem}_egoblur_temporal_report.json")

    report = run_egoblur_temporal_face_blur(
        input_video=args.input_video,
        output_video=args.output_video,
        face_model_path=args.face_model_path,
        egoblur_root=args.egoblur_root,
        output_json=output_json,
        camera_name=args.camera_name,
        config=FaceBlurConfig(
            detect_every_n_frames=args.detect_every_n_frames,
            face_score_threshold=args.face_model_score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            scale_factor_detections=args.scale_factor_detections,
            max_gap_frames=args.max_gap_frames,
            smooth_window=args.smooth_window,
            crf=args.crf,
            preset=args.preset,
        ),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "input_video": report["input_video"],
                "output_video": report["output_video"],
                "output_json": str(output_json),
                "detected_frame_count": report["detected_frame_count"],
                "tracked_frame_count": report["tracked_frame_count"],
                "track_count": report["track_count"],
                "elapsed_s": round(float(report["elapsed_s"]), 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
