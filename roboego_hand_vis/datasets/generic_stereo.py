from __future__ import annotations

from typing import Any

from .common import (
    ResolveConfig,
    as_abs,
    base_manifest,
    build_extrinsics_from_calibration,
    camera_intrinsics_from_json,
    probe_video,
    read_json,
    timestamps_info,
    tracking_info,
)


def resolve(config: ResolveConfig) -> dict[str, Any]:
    root = as_abs(config.root)
    if config.left_video is None or config.right_video is None or config.intrinsics_json is None:
        raise ValueError("generic_stereo requires --left_video, --right_video, and --intrinsics_json")
    left_video = as_abs(config.left_video, root)
    right_video = as_abs(config.right_video, root)
    intrinsics_path = as_abs(config.intrinsics_json, root)
    calib = read_json(intrinsics_path)
    left_info = probe_video(left_video)
    right_info = probe_video(right_video)
    return base_manifest(
        dataset_name="generic_stereo",
        sequence_name=config.sequence_name,
        root_dir=root,
        left_video=left_video,
        right_video=right_video,
        left_intrinsics=camera_intrinsics_from_json(calib, "left", left_info),
        right_intrinsics=camera_intrinsics_from_json(calib, "right", right_info),
        extrinsics=build_extrinsics_from_calibration(calib),
        camera_model={
            "type": "pinhole",
            "is_rectified": bool(config.assume_rectified),
            "is_undistorted": bool(config.assume_undistorted),
            "calibration_source": str(intrinsics_path),
        },
        tracking=tracking_info(as_abs(config.tracking_path, root) if config.tracking_path else None),
        timestamps=timestamps_info(
            as_abs(config.timestamps_left, root) if config.timestamps_left else None,
            as_abs(config.timestamps_right, root) if config.timestamps_right else None,
        ),
        annotations={},
        metadata={"adapter": "generic_stereo"},
    )
