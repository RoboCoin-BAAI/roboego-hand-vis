from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PALM_JOINT_INDICES = (0, 5, 9, 13, 17)


@dataclass(frozen=True)
class DepthAnchor:
    valid: bool
    depth_m: float | None
    source: str
    sample_count: int
    joint_indices: list[int]
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "depth_m": self.depth_m,
            "source": self.source,
            "sample_count": self.sample_count,
            "joint_indices": self.joint_indices,
            "reason": self.reason,
        }


def sample_depth_anchor(
    depth_m: np.ndarray,
    pixels_xy: np.ndarray,
    valid_pixels: np.ndarray,
    *,
    mode: str = "wrist_palm",
    window_radius_px: int = 4,
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.5,
) -> DepthAnchor:
    depth = np.asarray(depth_m, dtype=np.float32)
    pixels = np.asarray(pixels_xy, dtype=np.float32)
    valid = np.asarray(valid_pixels, dtype=bool)
    joint_indices = _anchor_joint_indices(mode)
    samples: list[np.ndarray] = []
    h, w = depth.shape[:2]
    radius = max(0, int(window_radius_px))

    for joint_idx in joint_indices:
        if joint_idx >= len(pixels) or joint_idx >= len(valid) or not bool(valid[joint_idx]):
            continue
        u, v = pixels[joint_idx]
        if not np.isfinite([u, v]).all():
            continue
        x = int(round(float(u)))
        y = int(round(float(v)))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        window = depth[y1:y2, x1:x2]
        ok = np.isfinite(window) & (window >= float(min_depth_m)) & (window <= float(max_depth_m))
        if ok.any():
            samples.append(window[ok])

    if not samples:
        return DepthAnchor(
            valid=False,
            depth_m=None,
            source=mode,
            sample_count=0,
            joint_indices=joint_indices,
            reason="no_valid_depth_samples",
        )
    values = np.concatenate(samples).astype(np.float32)
    return DepthAnchor(
        valid=True,
        depth_m=float(np.median(values)),
        source=mode,
        sample_count=int(len(values)),
        joint_indices=joint_indices,
    )


def fit_joints_xy_to_bbox(
    joints_camera: np.ndarray,
    *,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(joints_camera, dtype=np.float32)
    pixels = np.full((21, 2), np.nan, dtype=np.float32)
    valid = np.zeros((21,), dtype=bool)
    if joints.shape != (21, 3) or not np.isfinite(joints[:, :2]).all():
        return pixels, valid
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy[:4]]
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return pixels, valid

    xy = joints[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    extent = xy_max - xy_min
    if not np.isfinite(extent).all() or np.any(extent <= 1e-8):
        return pixels, valid
    norm = (xy - xy_min.reshape(1, 2)) / extent.reshape(1, 2)
    pixels[:, 0] = np.float32(x1) + norm[:, 0] * np.float32(x2 - x1)
    pixels[:, 1] = np.float32(y1) + norm[:, 1] * np.float32(y2 - y1)
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < int(width))
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < int(height))
    )
    return pixels, valid


def anchor_joints_with_depth(
    joints_camera: np.ndarray,
    *,
    target_depth_m: float,
    source_joint_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    joints = np.asarray(joints_camera, dtype=np.float32)
    if joints.shape != (21, 3) or not np.isfinite(joints).all():
        return joints.copy(), {"valid": False, "reason": "invalid_joints21"}
    source_z = float(joints[int(source_joint_index), 2])
    if not np.isfinite(source_z):
        return joints.copy(), {"valid": False, "reason": "invalid_source_depth"}
    dz = float(target_depth_m) - source_z
    anchored = joints.copy()
    anchored[:, 2] += np.float32(dz)
    return anchored, {
        "valid": True,
        "source_joint_index": int(source_joint_index),
        "source_depth_m": source_z,
        "target_depth_m": float(target_depth_m),
        "delta_z_m": dz,
    }


def anchor_joints_to_pixel_depth(
    joints_camera: np.ndarray,
    *,
    target_depth_m: float,
    target_pixel_xy: np.ndarray,
    intrinsics: dict[str, float],
    source_joint_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    joints = np.asarray(joints_camera, dtype=np.float32)
    if joints.shape != (21, 3) or not np.isfinite(joints).all():
        return joints.copy(), {"valid": False, "reason": "invalid_joints21"}
    pixel = np.asarray(target_pixel_xy, dtype=np.float32).reshape(2)
    if not np.isfinite(pixel).all() or not np.isfinite(float(target_depth_m)):
        return joints.copy(), {"valid": False, "reason": "invalid_depth_anchor"}
    z = float(target_depth_m)
    target = np.asarray(
        [
            (float(pixel[0]) - float(intrinsics["cx"])) * z / float(intrinsics["fx"]),
            (float(pixel[1]) - float(intrinsics["cy"])) * z / float(intrinsics["fy"]),
            z,
        ],
        dtype=np.float32,
    )
    source = joints[int(source_joint_index)]
    translation = target - source
    anchored = joints + translation.reshape(1, 3)
    return anchored.astype(np.float32), {
        "valid": True,
        "source_joint_index": int(source_joint_index),
        "target_pixel_xy": pixel.astype(float).tolist(),
        "source_joint_camera": source.astype(float).tolist(),
        "target_joint_camera": target.astype(float).tolist(),
        "target_depth_m": z,
        "translation": translation.astype(float).tolist(),
    }


def anchor_joints_by_projected_xy_and_depth(
    joints_camera: np.ndarray,
    *,
    target_depth_m: float,
    pixels_xy: np.ndarray,
    valid_pixels: np.ndarray,
    intrinsics: dict[str, float],
    source_joint_index: int = 0,
    min_depth_m: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    joints = np.asarray(joints_camera, dtype=np.float32)
    pixels = np.asarray(pixels_xy, dtype=np.float32)
    valid = np.asarray(valid_pixels, dtype=bool)
    if joints.shape != (21, 3) or not np.isfinite(joints).all():
        return joints.copy(), {"valid": False, "reason": "invalid_joints21"}
    if pixels.shape != (21, 2) or valid.shape != (21,) or not np.isfinite(pixels[valid]).all():
        return joints.copy(), {"valid": False, "reason": "invalid_projected_joints"}
    if not valid.all():
        return joints.copy(), {"valid": False, "reason": "projected_joint_outside_image"}

    source_idx = int(source_joint_index)
    source_z = float(joints[source_idx, 2])
    target_z = float(target_depth_m)
    if not np.isfinite([source_z, target_z]).all() or target_z <= float(min_depth_m):
        return joints.copy(), {"valid": False, "reason": "invalid_depth_anchor"}

    relative_z = joints[:, 2].astype(np.float32) - np.float32(source_z)
    anchored_z = relative_z + np.float32(target_z)
    if not np.isfinite(anchored_z).all() or np.any(anchored_z <= float(min_depth_m)):
        return joints.copy(), {"valid": False, "reason": "anchored_depth_out_of_range"}

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    anchored = np.empty((21, 3), dtype=np.float32)
    anchored[:, 2] = anchored_z
    anchored[:, 0] = (pixels[:, 0] - np.float32(cx)) * anchored_z / np.float32(fx)
    anchored[:, 1] = (pixels[:, 1] - np.float32(cy)) * anchored_z / np.float32(fy)
    return anchored, {
        "valid": True,
        "method": "preserve_projected_xy_and_anchor_relative_z",
        "source_joint_index": source_idx,
        "source_depth_m": source_z,
        "target_depth_m": target_z,
        "delta_z_m": target_z - source_z,
        "relative_z_min_m": float(np.min(relative_z)),
        "relative_z_max_m": float(np.max(relative_z)),
    }


def project_joints_to_image(
    joints_camera: np.ndarray,
    intrinsics: dict[str, float],
    *,
    width: int,
    height: int,
    min_z: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(joints_camera, dtype=np.float32)
    pixels = np.full((len(joints), 2), np.nan, dtype=np.float32)
    valid = np.zeros((len(joints),), dtype=bool)
    if joints.ndim != 2 or joints.shape[1] != 3:
        return pixels, valid
    z = joints[:, 2]
    ok = np.isfinite(joints).all(axis=1) & (z > float(min_z))
    pixels[ok, 0] = float(intrinsics["fx"]) * joints[ok, 0] / z[ok] + float(intrinsics["cx"])
    pixels[ok, 1] = float(intrinsics["fy"]) * joints[ok, 1] / z[ok] + float(intrinsics["cy"])
    valid = (
        ok
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < int(width))
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < int(height))
    )
    return pixels, valid


def _anchor_joint_indices(mode: str) -> list[int]:
    if mode == "wrist":
        return [0]
    if mode == "palm":
        return list(PALM_JOINT_INDICES)
    if mode == "wrist_palm":
        return list(PALM_JOINT_INDICES)
    raise ValueError(f"Unsupported depth anchor mode: {mode}")
