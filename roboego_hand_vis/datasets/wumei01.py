from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .common import ResolveConfig, as_abs, base_manifest, probe_video, timestamps_info, tracking_info


def _matrix_value(data: dict[str, Any], key: str) -> list[list[float]]:
    value = data.get(key)
    if value is None:
        raise KeyError(f"Missing '{key}' in rectify params")
    return [[float(item) for item in row] for row in value]


def _intrinsics_from_rectified_k(k: list[list[float]], video_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "fx": float(k[0][0]),
        "fy": float(k[1][1]),
        "cx": float(k[0][2]),
        "cy": float(k[1][2]),
        "width": int(video_info["width"]),
        "height": int(video_info["height"]),
        "distortion_model": None,
        "distortion_coeffs": [],
        "raw_extrinsics": None,
    }


def _load_rectify_params(path: Path) -> dict[str, Any]:
    with as_abs(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rectified = data.get("rectified") or {}
    stereo_params = data.get("stereo_params") or {}
    signed_baseline = stereo_params.get("baseline_m")
    if signed_baseline is None:
        p2 = np.asarray(_matrix_value(rectified, "P2"), dtype=float)
        fx = float(p2[0, 0])
        signed_baseline = float(p2[0, 3] / fx)
    baseline = abs(float(signed_baseline))
    return {
        "K1_rect": _matrix_value(rectified, "K1_rect"),
        "K2_rect": _matrix_value(rectified, "K2_rect"),
        "baseline_m": baseline,
        "raw_stereo_params": stereo_params,
        "raw_source_model": data.get("source_model"),
    }


def resolve(config: ResolveConfig) -> dict[str, Any]:
    root = as_abs(config.root)
    left_video = as_abs(config.left_video, root) if config.left_video else root / "videos" / "left_rectified.mp4"
    right_video = as_abs(config.right_video, root) if config.right_video else root / "videos" / "right_rectified.mp4"
    rectify_path = (
        as_abs(config.intrinsics_json, root)
        if config.intrinsics_json
        else root / "calibrations" / "rectify_params.yaml"
    )
    timestamps_left = (
        as_abs(config.timestamps_left, root) if config.timestamps_left else root / "timestamps" / "left_timestamps.txt"
    )
    timestamps_right = (
        as_abs(config.timestamps_right, root) if config.timestamps_right else root / "timestamps" / "right_timestamps.txt"
    )
    meta_path = root / "meta.json"
    tracking_path = as_abs(config.tracking_path, root) if config.tracking_path else None

    calib = _load_rectify_params(rectify_path)
    left_info = probe_video(left_video)
    right_info = probe_video(right_video)
    baseline = float(calib["baseline_m"])

    return base_manifest(
        dataset_name="wumei01",
        sequence_name=config.sequence_name,
        root_dir=root,
        left_video=left_video,
        right_video=right_video,
        left_intrinsics=_intrinsics_from_rectified_k(calib["K1_rect"], left_info),
        right_intrinsics=_intrinsics_from_rectified_k(calib["K2_rect"], right_info),
        extrinsics={
            "left_to_right": {
                "R": np.eye(3, dtype=float).tolist(),
                "t": [baseline, 0.0, 0.0],
                "baseline_m": baseline,
                "convention": "rectified_left_camera_to_right_camera",
            },
            "baseline_m": baseline,
            "raw": {"left": None, "right": None, "depth": None},
            "note": (
                "Wumei01 videos are already undistorted and stereo-rectified. "
                "The manifest uses positive baseline_m for downstream depth estimation; "
                "the signed source baseline is preserved in metadata.raw_stereo_params."
            ),
        },
        camera_model={
            "type": "pinhole",
            "is_rectified": bool(config.assume_rectified),
            "is_undistorted": bool(config.assume_undistorted),
            "calibration_source": str(rectify_path),
            "source_model": calib["raw_source_model"],
        },
        tracking=tracking_info(tracking_path if tracking_path and tracking_path.exists() else None),
        timestamps=timestamps_info(
            timestamps_left if timestamps_left.exists() else None,
            timestamps_right if timestamps_right.exists() else None,
        ),
        annotations={
            "meta_json": str(meta_path) if meta_path.exists() else None,
            "rectify_params_yaml": str(rectify_path),
        },
        metadata={
            "adapter": "wumei01",
            "raw_stereo_params": calib["raw_stereo_params"],
        },
    )
