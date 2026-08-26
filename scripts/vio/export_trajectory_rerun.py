#!/usr/bin/env python3
"""Export an OpenVINS trajectory NPZ as a standalone Rerun recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as trajectory:
        required = {"timestamp_camera", "frame_index", "T_world_camera"}
        missing = required.difference(trajectory.files)
        if missing:
            raise ValueError(
                f"Trajectory is missing arrays: {', '.join(sorted(missing))}"
            )
        timestamps = np.asarray(trajectory["timestamp_camera"], dtype=np.float64)
        frame_indices = np.asarray(trajectory["frame_index"], dtype=np.int64)
        transforms = np.asarray(trajectory["T_world_camera"], dtype=np.float64)
    if timestamps.ndim != 1 or frame_indices.shape != timestamps.shape:
        raise ValueError(
            "Trajectory timestamps and frame indices must be one-dimensional and aligned"
        )
    if transforms.shape != (len(timestamps), 4, 4):
        raise ValueError(
            f"Expected T_world_camera shape ({len(timestamps)}, 4, 4), got {transforms.shape}"
        )
    if not np.isfinite(transforms).all():
        raise ValueError("Trajectory contains non-finite transforms")
    return timestamps, frame_indices, transforms


def export_trajectory(trajectory_path: Path, output_path: Path) -> dict[str, object]:
    timestamps, frame_indices, transforms = load_trajectory(trajectory_path)
    if not len(timestamps):
        raise ValueError("Cannot export an empty trajectory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init("RoboEgo OpenVINS Trajectory", spawn=False)
    rr.save(str(output_path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/trajectory_convention",
        rr.TextDocument(
            "T_world_camera maps points from the raw left-camera frame into the OpenVINS world frame."
        ),
        static=True,
    )
    positions = transforms[:, :3, 3].astype(np.float32)
    for timestamp, frame_index, transform in zip(timestamps, frame_indices, transforms):
        rr.set_time("frame", sequence=int(frame_index))
        rr.set_time("camera_time", timestamp=float(timestamp))
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=transform[:3, 3],
                mat3x3=transform[:3, :3],
                axis_length=0.12,
            ),
        )
        rr.log(
            "world/current_camera_center",
            rr.Points3D([transform[:3, 3]], colors=[255, 190, 60], radii=0.018),
        )

    rr.log(
        "world/trajectory",
        rr.LineStrips3D([positions], colors=[80, 210, 255], radii=0.01),
        static=True,
    )
    rr.log(
        "world/pose_samples",
        rr.Points3D(positions, colors=[255, 255, 255], radii=0.008),
        static=True,
    )

    report = {
        "schema_version": "1.0",
        "trajectory_npz": str(trajectory_path),
        "output_rrd": str(output_path),
        "pose_count": len(timestamps),
        "first_frame_index": int(frame_indices[0]),
        "last_frame_index": int(frame_indices[-1]),
        "coordinate_convention": "T_world_camera",
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = export_trajectory(
        args.trajectory.expanduser().resolve(), args.output.expanduser().resolve()
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
