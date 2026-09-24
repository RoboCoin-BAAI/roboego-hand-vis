from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_DIR.parent.parent
for path in (PROJECT_ROOT, STAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from .prepare import prepare_openvins_run
except ImportError:
    from prepare import prepare_openvins_run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare raw session inputs and configuration for OpenVINS.")
    parser.add_argument("--session", required=True, help="输入 session 目录。")
    parser.add_argument("--output", required=True, help="VIO 输出目录。")
    parser.add_argument("--input_mode", choices=("raw", "rectified"), default="raw", help="VIO 输入模式：raw 使用原始视频，rectified 使用去畸变/立体矫正视频。")
    parser.add_argument("--stereo_tolerance_s", type=float, default=0.001)
    parser.add_argument("--max_imu_gap_s", type=float, default=0.05)
    parser.add_argument("--max_imu_boundary_gap_s", type=float, default=0.1)
    parser.add_argument("--downsample_cameras", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = prepare_openvins_run(
        args.session,
        args.output,
        input_mode=args.input_mode,
        tolerance_s=args.stereo_tolerance_s,
        downsample_cameras=args.downsample_cameras,
        max_imu_gap_s=args.max_imu_gap_s,
        max_imu_boundary_gap_s=args.max_imu_boundary_gap_s,
    )
    print(json.dumps({"manifest_path": str(Path(result.manifest_path))}, indent=2))


if __name__ == "__main__":
    main()
