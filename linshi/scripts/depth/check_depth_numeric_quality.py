from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.depth_numeric_quality.logic import DepthNumericQualityConfig, run_depth_numeric_quality_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight numeric quality checks on saved depth .npy files.")
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=4.0)
    parser.add_argument("--min_valid_depth_ratio", type=float, default=0.5)
    parser.add_argument("--min_depth_std_m", type=float, default=0.02)
    parser.add_argument("--min_depth_iqr_m", type=float, default=0.03)
    parser.add_argument("--max_near_ratio", type=float, default=0.8)
    parser.add_argument("--max_far_ratio", type=float, default=0.8)
    parser.add_argument("--edge_band_m", type=float, default=0.05)
    parser.add_argument("--max_median_delta_m", type=float, default=0.5)
    parser.add_argument("--min_valid_mask_iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_depth_numeric_quality_check(
        depth_dir=Path(args.depth_dir),
        output_dir=Path(args.output_dir),
        config=DepthNumericQualityConfig(
            start=args.start,
            num_frames=args.num_frames,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            min_valid_depth_ratio=args.min_valid_depth_ratio,
            min_depth_std_m=args.min_depth_std_m,
            min_depth_iqr_m=args.min_depth_iqr_m,
            max_near_ratio=args.max_near_ratio,
            max_far_ratio=args.max_far_ratio,
            edge_band_m=args.edge_band_m,
            max_median_delta_m=args.max_median_delta_m,
            min_valid_mask_iou=args.min_valid_mask_iou,
        ),
    )
    output_dir = Path(args.output_dir)
    print(f"Depth numeric quality frames: {report['summary']['frame_count']}")
    print(f"Warning frames: {report['summary']['warning_frame_count']}")
    print(f"Bad segments: {len(report['bad_segments'])}")
    print(f"Report: {output_dir / 'depth_numeric_quality_report.json'}")
    print(f"CSV: {output_dir / 'depth_numeric_quality_frames.csv'}")


if __name__ == "__main__":
    main()
