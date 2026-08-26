#!/usr/bin/env python3
"""Prepare raw session inputs and configuration for OpenVINS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roboego_hand_vis.vio.prepare import prepare_openvins_run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session", required=True, help="Raw RoboEgo session directory."
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for this OpenVINS run."
    )
    parser.add_argument("--stereo_tolerance_s", type=float, default=0.001)
    parser.add_argument("--max_imu_gap_s", type=float, default=0.05)
    parser.add_argument("--max_imu_boundary_gap_s", type=float, default=0.1)
    parser.add_argument(
        "--downsample_cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let OpenVINS halve raw image resolution before tracking.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = prepare_openvins_run(
        args.session,
        args.output,
        tolerance_s=args.stereo_tolerance_s,
        downsample_cameras=args.downsample_cameras,
        max_imu_gap_s=args.max_imu_gap_s,
        max_imu_boundary_gap_s=args.max_imu_boundary_gap_s,
    )
    print(json.dumps({"manifest_path": str(Path(result.manifest_path))}, indent=2))


if __name__ == "__main__":
    main()
