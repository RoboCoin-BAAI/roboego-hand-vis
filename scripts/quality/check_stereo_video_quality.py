from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.stereo_video import SummaryQualityConfig, run_summary_quality_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample stereo videos from dataset_manifest.json and check quality.")
    parser.add_argument("--manifest", required=True, help="Path to dataset_manifest.json.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_count", type=int, default=12)
    parser.add_argument("--dark_threshold", type=int, default=10)
    parser.add_argument("--bright_threshold", type=int, default=245)
    parser.add_argument("--orb_features", type=int, default=1500)
    parser.add_argument("--max_brightness_gap", type=float, default=35.0)
    parser.add_argument("--min_sharpness", type=float, default=50.0)
    parser.add_argument("--max_dark_ratio", type=float, default=0.25)
    parser.add_argument("--max_bright_ratio", type=float, default=0.25)
    parser.add_argument("--min_orb_matches", type=int, default=30)
    parser.add_argument("--max_median_abs_dy", type=float, default=2.0)
    parser.add_argument("--max_p90_abs_dy", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SummaryQualityConfig(
        sample_count=args.sample_count,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        orb_features=args.orb_features,
        max_brightness_gap=args.max_brightness_gap,
        min_sharpness=args.min_sharpness,
        max_dark_ratio=args.max_dark_ratio,
        max_bright_ratio=args.max_bright_ratio,
        min_orb_matches=args.min_orb_matches,
        max_median_abs_dy=args.max_median_abs_dy,
        max_p90_abs_dy=args.max_p90_abs_dy,
    )
    report = run_summary_quality_check(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        config=config,
    )
    output_dir = Path(args.output_dir).expanduser()
    status = "PASS" if report["summary"]["passed"] else "FAIL"
    print(f"Stereo quality: {status}")
    print(f"JSON: {output_dir / 'stereo_quality_report.json'}")
    print(f"CSV: {output_dir / 'stereo_quality_samples.csv'}")
    print(f"Contact sheet: {output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
