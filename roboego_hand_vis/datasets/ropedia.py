from __future__ import annotations

from typing import Any

import numpy as np

from .common import (
    ResolveConfig,
    as_abs,
    base_manifest,
    build_default_extrinsics,
    probe_video,
    timestamps_info,
    tracking_info,
)


ROPEDIA_INTRINSICS = {
    "fx": 200.0,
    "fy": 200.0,
    "cx": 256.0,
    "cy": 256.0,
}


def ropedia_intrinsics(
    video_info: dict[str, Any] | None,
    k_values: list[float] | tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    fx, fy, cx, cy = k_values if k_values is not None else (
        ROPEDIA_INTRINSICS["fx"],
        ROPEDIA_INTRINSICS["fy"],
        ROPEDIA_INTRINSICS["cx"],
        ROPEDIA_INTRINSICS["cy"],
    )
    out: dict[str, Any] = {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
    out.update(
        {
            "width": int(video_info["width"]) if video_info else None,
            "height": int(video_info["height"]) if video_info else None,
            "distortion_model": None,
            "distortion_coeffs": [],
            "raw_extrinsics": None,
        }
    )
    return out


def load_stereo_calibration(annotation_path) -> dict[str, Any]:
    if not annotation_path.exists():
        return {
            "K": None,
            "baseline_m": None,
            "left_to_right": None,
            "calibration_source": "ROPEDIA_K_DEFAULT",
        }

    import h5py

    with h5py.File(annotation_path, "r") as f:
        cam01 = f.get("calibration/cam01")
        if cam01 is None or "K" not in cam01:
            return {
                "K": None,
                "baseline_m": None,
                "left_to_right": None,
                "calibration_source": "ROPEDIA_K_DEFAULT",
            }
        k_values = [float(v) for v in cam01["K"][()]]
        baseline = float(cam01["baseline"][()]) if "baseline" in cam01 else None

    left_to_right = None
    if baseline is not None:
        left_to_right = {
            "R": np.eye(3, dtype=float).tolist(),
            "t": [baseline, 0.0, 0.0],
            "baseline_m": baseline,
            "convention": "rectified_left_camera_to_right_camera",
        }
    return {
        "K": k_values,
        "baseline_m": baseline,
        "left_to_right": left_to_right,
        "calibration_source": str(annotation_path),
    }


def resolve(config: ResolveConfig) -> dict[str, Any]:
    root = as_abs(config.root)
    left_video = as_abs(config.left_video, root) if config.left_video else as_abs(root / "stereo_left.mp4")
    right_video = as_abs(config.right_video, root) if config.right_video else as_abs(root / "stereo_right.mp4")
    left_info = probe_video(left_video)
    right_info = probe_video(right_video)
    annotation_path = root / "annotation.hdf5"
    tracking_path = as_abs(config.tracking_path, root) if config.tracking_path else root / "tracking.ndjson"
    stereo_calibration = load_stereo_calibration(annotation_path)
    return base_manifest(
        dataset_name="ropedia",
        sequence_name=config.sequence_name,
        root_dir=root,
        left_video=left_video,
        right_video=right_video,
        left_intrinsics=ropedia_intrinsics(left_info, stereo_calibration["K"]),
        right_intrinsics=ropedia_intrinsics(right_info, stereo_calibration["K"]),
        extrinsics={
            **build_default_extrinsics(),
            "left_to_right": stereo_calibration["left_to_right"],
            "baseline_m": stereo_calibration["baseline_m"],
        },
        camera_model={
            "type": "pinhole",
            "is_rectified": bool(config.assume_rectified),
            "is_undistorted": bool(config.assume_undistorted),
            "calibration_source": stereo_calibration["calibration_source"],
        },
        tracking=tracking_info(tracking_path if as_abs(tracking_path).exists() else None),
        timestamps=timestamps_info(
            as_abs(config.timestamps_left, root) if config.timestamps_left else None,
            as_abs(config.timestamps_right, root) if config.timestamps_right else None,
        ),
        annotations={
            "annotation_hdf5": str(annotation_path) if annotation_path.exists() else None,
            "official_hand_mocap_available": annotation_path.exists(),
        },
        metadata={"adapter": "ropedia"},
    )
