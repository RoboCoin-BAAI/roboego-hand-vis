from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_STATUS_WEIGHTS = {
    "normal": 1.0,
    "detected": 1.0,
    "recovered": 0.6,
    "smooth": 0.3,
    "interpolated": 0.3,
}

DEFAULT_MAX_CORRECTIONS_M = {
    "normal": 0.015,
    "detected": 0.015,
    "recovered": 0.03,
    "smooth": 0.05,
    "interpolated": 0.05,
}

DEFAULT_MAX_ROTATION_DEGREES = {
    "normal": 5.0,
    "detected": 5.0,
    "recovered": 10.0,
    "smooth": 15.0,
    "interpolated": 15.0,
}


def status_weight(status: str, weights: Dict[str, float] | None = None) -> float:
    table = DEFAULT_STATUS_WEIGHTS if weights is None else weights
    return float(table.get(status, table.get("normal", 1.0)))


def correction_limit_m(status: str, limits: Dict[str, float] | None = None) -> float:
    table = DEFAULT_MAX_CORRECTIONS_M if limits is None else limits
    return float(table.get(status, table.get("normal", 0.015)))


def rotation_limit_degrees(status: str, limits: Dict[str, float] | None = None) -> float:
    table = DEFAULT_MAX_ROTATION_DEGREES if limits is None else limits
    return float(table.get(status, table.get("normal", 5.0)))


def clamp_vector_delta(current: np.ndarray, target: np.ndarray, max_norm: float) -> Tuple[np.ndarray, float]:
    delta = target.astype(np.float32) - current.astype(np.float32)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-9 or norm <= max_norm:
        return target.astype(np.float32), norm
    return (current + delta * (float(max_norm) / norm)).astype(np.float32), float(max_norm)


def smooth_centers_weighted(
    centers: Iterable[Iterable[float]],
    statuses: Iterable[str],
    *,
    window: int = 5,
    weights: Dict[str, float] | None = None,
    max_corrections_m: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    center_arr = np.asarray(list(centers), dtype=np.float32)
    status_list = list(statuses)
    if len(center_arr) != len(status_list):
        raise ValueError("centers and statuses must have the same length")
    if len(center_arr) == 0:
        return []

    half = max(0, int(window) // 2)
    out: List[Dict[str, Any]] = []
    for idx, center in enumerate(center_arr):
        start = max(0, idx - half)
        end = min(len(center_arr), idx + half + 1)
        local = center_arr[start:end]
        local_statuses = status_list[start:end]
        local_weights = np.asarray([status_weight(s, weights) for s in local_statuses], dtype=np.float32)
        if float(local_weights.sum()) <= 1e-9:
            target = center
        else:
            target = (local * local_weights[:, None]).sum(axis=0) / local_weights.sum()
        smoothed, correction_norm = clamp_vector_delta(
            center,
            target,
            correction_limit_m(status_list[idx], max_corrections_m),
        )
        out.append(
            {
                "center": smoothed.astype(float).tolist(),
                "raw_center": center.astype(float).tolist(),
                "target_center": target.astype(float).tolist(),
                "status": status_list[idx],
                "correction_norm_m": correction_norm,
                "window_start": int(start),
                "window_end": int(end - 1),
            }
        )
    return out


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation.astype(np.float64)))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def axis_angle_to_rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12 or abs(degrees) <= 1e-12:
        return np.eye(3, dtype=np.float32)
    axis = axis / norm
    x, y, z = axis
    theta = np.deg2rad(float(degrees))
    c = np.cos(theta)
    s = np.sin(theta)
    c1 = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float32,
    )
    return rotation


def rotation_axis(rotation: np.ndarray) -> np.ndarray:
    angle_rad = np.deg2rad(rotation_angle_degrees(rotation))
    if abs(angle_rad) <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(angle_rad)
    return axis.astype(np.float32)


def clamp_rotation(rotation: np.ndarray, max_degrees: float) -> Tuple[np.ndarray, float]:
    angle = rotation_angle_degrees(rotation)
    if angle <= max_degrees:
        return rotation.astype(np.float32), angle
    axis = rotation_axis(rotation)
    return axis_angle_to_rotation(axis, max_degrees), float(max_degrees)


def kabsch_rotation(
    source_points: Iterable[Iterable[float]],
    target_points: Iterable[Iterable[float]],
) -> Tuple[np.ndarray, float]:
    source = np.asarray(list(source_points), dtype=np.float32)
    target = np.asarray(list(target_points), dtype=np.float32)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape")
    if len(source) < 3:
        return np.eye(3, dtype=np.float32), 0.0

    source_centered = source - np.mean(source, axis=0, keepdims=True)
    target_centered = target - np.mean(target, axis=0, keepdims=True)
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    rotation = rotation.astype(np.float32)
    return rotation, rotation_angle_degrees(rotation)


def smooth_vertices_rigid(
    vertices_by_frame: List[np.ndarray],
    statuses: List[str],
    *,
    window: int = 5,
    max_rotation_degrees: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    if len(vertices_by_frame) != len(statuses):
        raise ValueError("vertices_by_frame and statuses must have the same length")
    if not vertices_by_frame:
        return []
    half = max(0, int(window) // 2)
    out: List[Dict[str, Any]] = []
    for idx, vertices in enumerate(vertices_by_frame):
        start = max(0, idx - half)
        end = min(len(vertices_by_frame), idx + half + 1)
        local_weights = np.asarray([status_weight(s) for s in statuses[start:end]], dtype=np.float32)
        local_centered = []
        for local_vertices in vertices_by_frame[start:end]:
            center = np.median(local_vertices, axis=0, keepdims=True)
            local_centered.append(local_vertices.astype(np.float32) - center)
        reference = np.average(np.stack(local_centered, axis=0), axis=0, weights=local_weights)
        current_center = np.median(vertices, axis=0, keepdims=True)
        current_centered = vertices.astype(np.float32) - current_center
        rotation, raw_angle = kabsch_rotation(current_centered, reference)
        rotation, applied_angle = clamp_rotation(
            rotation,
            rotation_limit_degrees(statuses[idx], max_rotation_degrees),
        )
        rotated_centered = current_centered @ rotation.T
        out.append(
            {
                "vertices_centered": rotated_centered.astype(np.float32),
                "rotation": rotation.astype(float).tolist(),
                "raw_angle_degrees": float(raw_angle),
                "applied_angle_degrees": float(applied_angle),
                "window_start": int(start),
                "window_end": int(end - 1),
            }
        )
    return out
