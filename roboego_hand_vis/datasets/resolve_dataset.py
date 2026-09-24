from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.datasets.manifest import ResolveConfig, resolve_dataset_manifest, write_manifest


def optional_path(value: str) -> Path | None:
    return Path(value).expanduser() if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a raw ego-stereo dataset into a standard dataset_manifest.json."
    )
    parser.add_argument("--dataset", choices=["raw", "wuxi", "ropedia", "generic_stereo"], required=True)
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for dataset_manifest.json.")
    parser.add_argument("--sequence_name", default="default")
    parser.add_argument("--left_video", default="", help="Override left video path.")
    parser.add_argument("--right_video", default="", help="Override right video path.")
    parser.add_argument("--intrinsics_json", default="", help="Override camera intrinsics JSON path.")
    parser.add_argument("--tracking_path", default="", help="Optional tracking/head pose/IMU file.")
    parser.add_argument("--timestamps_left", default="", help="Optional left frame timestamp file.")
    parser.add_argument("--timestamps_right", default="", help="Optional right frame timestamp file.")
    parser.add_argument("--assume_rectified", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assume_undistorted", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ResolveConfig(
        dataset=args.dataset,
        root=Path(args.root).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        sequence_name=args.sequence_name,
        left_video=optional_path(args.left_video),
        right_video=optional_path(args.right_video),
        intrinsics_json=optional_path(args.intrinsics_json),
        tracking_path=optional_path(args.tracking_path),
        timestamps_left=optional_path(args.timestamps_left),
        timestamps_right=optional_path(args.timestamps_right),
        assume_rectified=args.assume_rectified,
        assume_undistorted=args.assume_undistorted,
    )
    manifest = resolve_dataset_manifest(config)
    manifest_path, summary_path = write_manifest(manifest, config.output_dir)
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
