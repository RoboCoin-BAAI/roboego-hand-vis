#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.video_segments import DEFAULT_CAMERA_STREAMS, export_selected_video_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut selected source videos from a semantic processing plan.")
    parser.add_argument("--session", required=True, help="Source session directory containing videos/*.mp4 and meta.json.")
    parser.add_argument("--processing_plan", required=True, help="processing_plan_*.json from build_processing_plan.py.")
    parser.add_argument("--output_dir", required=True, help="Directory for selected segment folders.")
    parser.add_argument("--camera_stream", action="append", choices=("left", "right", "bleft", "bright"), default=None)
    parser.add_argument("--fps", type=float, default=None, help="Override fps. Defaults to session meta.json fps.")
    parser.add_argument("--codec", default="h264", help="Default h264 uses ffmpeg/libx264 for browser-compatible MP4; pass mp4v for legacy OpenCV output.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    streams = tuple(args.camera_stream) if args.camera_stream else DEFAULT_CAMERA_STREAMS
    summary = export_selected_video_segments(
        session_dir=Path(args.session).expanduser(),
        processing_plan_json=Path(args.processing_plan).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        camera_streams=streams,
        fps=args.fps,
        overwrite=args.overwrite,
        codec=args.codec,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
