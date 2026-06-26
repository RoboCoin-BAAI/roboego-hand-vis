from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ResolveConfig:
    dataset: str
    root: Path
    output_dir: Path
    sequence_name: str = "default"
    left_video: Path | None = None
    right_video: Path | None = None
    intrinsics_json: Path | None = None
    tracking_path: Path | None = None
    timestamps_left: Path | None = None
    timestamps_right: Path | None = None
    assume_rectified: bool = True
    assume_undistorted: bool = True


def as_abs(path: Path, base: Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base is not None:
        return (base / path).resolve()
    return path.resolve()


def probe_video(path: Path) -> dict[str, Any]:
    path = as_abs(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    duration_sec = float(frame_count / fps) if fps > 0 else None
    cap.release()

    return {
        "path": str(path),
        "exists": path.exists(),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def read_json(path: Path) -> dict[str, Any]:
    with as_abs(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def camera_intrinsics_from_json(
    calib: dict[str, Any],
    camera: str,
    video_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if camera not in calib:
        raise KeyError(f"Camera '{camera}' not found in intrinsics JSON. Available: {sorted(calib)}")
    src = calib[camera]
    return {
        "fx": float(src["fx"]),
        "fy": float(src["fy"]),
        "cx": float(src["cx"]),
        "cy": float(src["cy"]),
        "width": int(video_info["width"]) if video_info else src.get("width"),
        "height": int(video_info["height"]) if video_info else src.get("height"),
        "distortion_model": src.get("distortion_model"),
        "distortion_coeffs": list(src.get("distortion", src.get("distortion_coeffs", []))),
        "raw_extrinsics": src.get("extrinsics"),
    }


def build_extrinsics_from_calibration(calib: dict[str, Any]) -> dict[str, Any]:
    baseline = calib.get("baseline_m")
    out: dict[str, Any] = {
        "left_to_right": None,
        "baseline_m": float(baseline) if baseline is not None else None,
        "raw": {
            "left": calib.get("left", {}).get("extrinsics"),
            "right": calib.get("right", {}).get("extrinsics"),
            "depth": calib.get("depth", {}).get("extrinsics"),
        },
        "note": (
            "left_to_right uses rectified stereo convention when baseline_m exists. "
            "Raw camera extrinsics are preserved separately because source convention may differ."
        ),
    }
    if baseline is not None:
        out["left_to_right"] = {
            "R": np.eye(3, dtype=float).tolist(),
            "t": [float(baseline), 0.0, 0.0],
            "baseline_m": float(baseline),
            "convention": "rectified_left_camera_to_right_camera",
        }
    return out


def build_default_extrinsics() -> dict[str, Any]:
    return {
        "left_to_right": None,
        "baseline_m": None,
        "raw": {"left": None, "right": None, "depth": None},
        "note": "No stereo extrinsics were found by the dataset adapter.",
    }


def tracking_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "tracking_path": None,
            "head_pose_available": False,
            "imu_available": False,
            "hand_tracking_available": False,
            "coordinate_frame": None,
        }
    path = as_abs(path)
    first: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
            first = json.loads(line) if line else {}
    return {
        "tracking_path": str(path),
        "head_pose_available": "head_6dof" in first,
        "imu_available": "imu" in first,
        "hand_tracking_available": "left_hand_joints" in first or "right_hand_joints" in first,
        "coordinate_frame": "source_tracking_frame",
        "sample_keys": sorted(first.keys()) if first else [],
    }


def timestamps_info(left_path: Path | None, right_path: Path | None) -> dict[str, Any]:
    return {
        "type": "file" if left_path or right_path else "frame_index",
        "left_path": str(as_abs(left_path)) if left_path else None,
        "right_path": str(as_abs(right_path)) if right_path else None,
    }


def base_manifest(
    *,
    dataset_name: str,
    sequence_name: str,
    root_dir: Path,
    left_video: Path,
    right_video: Path,
    left_intrinsics: dict[str, Any],
    right_intrinsics: dict[str, Any],
    extrinsics: dict[str, Any],
    camera_model: dict[str, Any],
    tracking: dict[str, Any],
    timestamps: dict[str, Any],
    annotations: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_info = probe_video(left_video)
    right_info = probe_video(right_video)
    return {
        "schema_version": "1.0",
        "dataset_name": dataset_name,
        "sequence_name": sequence_name,
        "root_dir": str(as_abs(root_dir)),
        "reference_camera": "left",
        "videos": {
            "left": left_info,
            "right": right_info,
        },
        "camera_model": camera_model,
        "intrinsics": {
            "left": left_intrinsics,
            "right": right_intrinsics,
        },
        "extrinsics": extrinsics,
        "timestamps": timestamps,
        "trajectory": tracking,
        "annotations": annotations or {},
        "metadata": metadata or {},
    }


def manifest_summary(manifest: dict[str, Any]) -> str:
    left = manifest["videos"]["left"]
    right = manifest["videos"]["right"]
    intr = manifest["intrinsics"]["left"]
    extr = manifest.get("extrinsics", {})
    traj = manifest.get("trajectory", {})
    lines = [
        f"dataset: {manifest['dataset_name']}",
        f"sequence: {manifest['sequence_name']}",
        f"root: {manifest['root_dir']}",
        f"left_video: {left['path']}",
        f"right_video: {right['path']}",
        f"resolution: left={left['width']}x{left['height']} right={right['width']}x{right['height']}",
        f"fps: left={left['fps']:.6g} right={right['fps']:.6g}",
        f"frame_count: left={left['frame_count']} right={right['frame_count']}",
        f"left_intrinsics: fx={intr['fx']:.6g} fy={intr['fy']:.6g} cx={intr['cx']:.6g} cy={intr['cy']:.6g}",
        f"baseline_m: {extr.get('baseline_m')}",
        f"camera_model: {manifest['camera_model']}",
        f"tracking_path: {traj.get('tracking_path')}",
        f"head_pose_available: {traj.get('head_pose_available')}",
        f"imu_available: {traj.get('imu_available')}",
        f"hand_tracking_available: {traj.get('hand_tracking_available')}",
    ]
    return "\n".join(lines) + "\n"
