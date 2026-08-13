#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.datasets.crop_manifest import (
    crop_video,
    roi_from_ratios,
    update_manifest_for_crop,
    validate_roi,
    write_cropped_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop rectified stereo videos and write an updated dataset manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--x_min_ratio", type=float, default=0.15)
    parser.add_argument("--x_max_ratio", type=float, default=0.85)
    parser.add_argument("--y_min_ratio", type=float, default=0.35)
    parser.add_argument("--y_max_ratio", type=float, default=1.0)
    parser.add_argument("--roi_xyxy", nargs=4, type=int, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    videos_dir = output_dir / "videos"
    manifest_dir = output_dir / "manifest"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    left_src = Path(manifest["videos"]["left"]["path"]).expanduser().resolve()
    right_src = Path(manifest["videos"]["right"]["path"]).expanduser().resolve()
    width = int(manifest["videos"]["left"]["width"])
    height = int(manifest["videos"]["left"]["height"])
    if args.roi_xyxy is not None:
        roi = validate_roi(tuple(args.roi_xyxy), width=width, height=height)
    else:
        roi = roi_from_ratios(
            width=width,
            height=height,
            x_min=args.x_min_ratio,
            x_max=args.x_max_ratio,
            y_min=args.y_min_ratio,
            y_max=args.y_max_ratio,
        )

    left_out = videos_dir / "left_rectified_crop.mp4"
    right_out = videos_dir / "right_rectified_crop.mp4"
    left_summary = crop_video(left_src, left_out, roi, codec=args.codec, overwrite=args.overwrite)
    right_summary = crop_video(right_src, right_out, roi, codec=args.codec, overwrite=args.overwrite)
    cropped_manifest = update_manifest_for_crop(manifest, left_video=left_out, right_video=right_out, roi=roi)
    manifest_out, summary_out = write_cropped_manifest(cropped_manifest, manifest_dir)

    print(
        json.dumps(
            {
                "roi_xyxy": [int(v) for v in roi],
                "left_video": left_summary,
                "right_video": right_summary,
                "manifest": str(manifest_out),
                "summary": str(summary_out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
