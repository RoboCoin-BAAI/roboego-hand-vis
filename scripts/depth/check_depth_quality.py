from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.depth.quality import DepthQualityConfig, run_depth_quality_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FoundationStereo depth quality using dataset_manifest.json.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stereo_quality_report", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=4.0)
    parser.add_argument("--point_stride", type=int, default=4)
    parser.add_argument("--max_preview_points", type=int, default=90000)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--warp_sample_stride", type=int, default=2)
    parser.add_argument("--edge_percentile", type=float, default=85.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DepthQualityConfig(
        start=args.start,
        num_frames=args.num_frames,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        point_stride=args.point_stride,
        max_preview_points=args.max_preview_points,
        overlay_alpha=args.overlay_alpha,
        warp_sample_stride=args.warp_sample_stride,
        edge_percentile=args.edge_percentile,
    )
    report = run_depth_quality_check(
        manifest_path=Path(args.manifest),
        depth_dir=Path(args.depth_dir),
        output_dir=Path(args.output_dir),
        config=config,
        stereo_quality_report=Path(args.stereo_quality_report) if args.stereo_quality_report else None,
    )
    output_dir = Path(args.output_dir)
    print(f"Depth quality frames: {report['summary']['frame_count']}")
    print(f"Warning frames: {report['summary']['warning_frame_count']}")
    print(f"Report: {output_dir / 'depth_quality_report.json'}")
    print(f"CSV: {output_dir / 'depth_quality_frames.csv'}")
    print(f"Vis: {output_dir / 'vis'}")
    print(f"Point cloud preview: {output_dir / 'pointcloud_preview'}")


if __name__ == "__main__":
    main()
