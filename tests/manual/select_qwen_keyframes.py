#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.manual.qwen_keyframe_selector_lib import KeyframeSelectorConfig, select_qwen_keyframes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Qwen keyframes from video simple quality intervals.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--video_simple_quality_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--preview_dir", default="")
    parser.add_argument("--analysis_fps", type=float, default=2.0)
    parser.add_argument("--base_interval_s", type=float, default=5.0)
    parser.add_argument("--boundary_margin_s", type=float, default=0.0)
    parser.add_argument("--min_keyframe_gap_s", type=float, default=1.0)
    parser.add_argument("--visual_change_quantile", type=float, default=0.85)
    parser.add_argument("--max_keyframes_per_minute", type=int, default=30)
    parser.add_argument("--max_width", type=int, default=320)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = select_qwen_keyframes(
        video_path=Path(args.video),
        video_simple_quality_json=Path(args.video_simple_quality_json),
        output_json=Path(args.output_json),
        config=KeyframeSelectorConfig(
            analysis_fps=args.analysis_fps,
            base_interval_s=args.base_interval_s,
            boundary_margin_s=args.boundary_margin_s,
            min_keyframe_gap_s=args.min_keyframe_gap_s,
            visual_change_quantile=args.visual_change_quantile,
            max_keyframes_per_minute=args.max_keyframes_per_minute,
            max_width=args.max_width,
        ),
    )
    if args.preview_dir:
        write_preview_frames(Path(args.video), Path(args.preview_dir), report["keyframes"])
    print(
        json.dumps(
            {
                "output_json": str(Path(args.output_json)),
                "keyframe_count": report["keyframe_count"],
                "source_interval_count": len(report["source_intervals"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def write_preview_frames(video_path: Path, preview_dir: Path, keyframes: list[dict]) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    preview_dir.mkdir(parents=True, exist_ok=True)
    try:
        for idx, keyframe in enumerate(keyframes):
            time_s = float(keyframe["t"])
            frame_idx = int(round(time_s * fps)) if fps > 0 else 0
            frame_idx = max(0, min(frame_count - 1, frame_idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            label = "_".join(str(reason) for reason in keyframe.get("reasons", []))[:80]
            path = preview_dir / f"keyframe_{idx:04d}_t{time_s:09.3f}_{label}.jpg"
            cv2.imwrite(str(path), frame)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
