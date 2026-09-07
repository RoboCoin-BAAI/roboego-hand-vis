#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.depth.archive import DepthArchiveConfig, archive_depth_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive float32 depth npy cache as uint16 PNG and FFV1 AVI.")
    parser.add_argument("--depth_dir", required=True, help="Directory containing frame_*_foundation_depth_m.npy files.")
    parser.add_argument("--output_dir", required=True, help="Archive output directory.")
    parser.add_argument("--max_depth_m", type=float, default=4.0)
    parser.add_argument("--invalid_value", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ffmpeg_bin", default="ffmpeg")
    parser.add_argument("--write_png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write_avi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write_preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview_mode", choices=["near", "full"], default="near")
    parser.add_argument("--preview_near_min_m", type=float, default=0.15)
    parser.add_argument("--preview_near_max_m", type=float, default=3.0)
    parser.add_argument(
        "--preview_contrast_percentile",
        type=float,
        default=2.0,
        help="Percentile clipping for color preview contrast. Use 0 to map by max_depth_m only.",
    )
    parser.add_argument("--preview_codec", default="mp4v")
    parser.add_argument("--ffv1_container", choices=["avi", "mkv"], default="avi")
    parser.add_argument("--dry_run_avi", action="store_true", help="Write PNG/meta but only record the ffmpeg command.")
    parser.add_argument("--delete_npy", action="store_true", help="Delete source npy files after successful archive.")
    parser.add_argument("--delete_png", action="store_true", help="Delete temporary uint16 PNG frames after AVI is written.")
    parser.add_argument("--write_hevc12", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hevc12_crf", type=int, default=0)
    parser.add_argument("--hevc12_preset", default="slow")
    parser.add_argument("--hevc12_lossless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write_valid_mask", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = archive_depth_cache(
        depth_dir=Path(args.depth_dir),
        output_dir=Path(args.output_dir),
        config=DepthArchiveConfig(
            max_depth_m=args.max_depth_m,
            invalid_value=args.invalid_value,
            fps=args.fps,
            write_png=args.write_png,
            write_avi=args.write_avi,
            write_preview=args.write_preview,
            preview_mode=args.preview_mode,
            preview_near_min_m=args.preview_near_min_m,
            preview_near_max_m=args.preview_near_max_m,
            preview_contrast_percentile=args.preview_contrast_percentile or None,
            preview_codec=args.preview_codec,
            ffv1_container=args.ffv1_container,
            dry_run_avi=args.dry_run_avi,
            delete_npy=args.delete_npy,
            delete_png=args.delete_png,
            ffmpeg_bin=args.ffmpeg_bin,
            write_hevc12=args.write_hevc12,
            hevc12_crf=args.hevc12_crf,
            hevc12_preset=args.hevc12_preset,
            hevc12_lossless=args.hevc12_lossless,
            write_valid_mask=args.write_valid_mask,
        ),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
