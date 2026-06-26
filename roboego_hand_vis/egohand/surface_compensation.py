from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np


def face_normals_and_centers(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    verts = vertices.astype(np.float32)
    tri = faces.astype(np.int64)
    p0 = verts[tri[:, 0]]
    p1 = verts[tri[:, 1]]
    p2 = verts[tri[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    centers = (p0 + p1 + p2) / 3.0
    return normals.astype(np.float32), centers.astype(np.float32)


def camera_visible_face_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    dot_threshold: float = 0.0,
) -> np.ndarray:
    """Return faces whose outward normal points toward the camera origin.

    The input vertices are in OpenCV camera coordinates, where points in front
    of the camera have positive z. The view direction from a surface point to
    the camera is approximately ``-center``.
    """
    normals, centers = face_normals_and_centers(vertices, faces)
    view_to_camera = -centers
    view_norm = np.linalg.norm(view_to_camera, axis=1, keepdims=True)
    view_to_camera = view_to_camera / np.maximum(view_norm, 1e-12)
    facing = np.sum(normals * view_to_camera, axis=1)
    return facing > float(dot_threshold)


def visible_vertex_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    dot_threshold: float = 0.0,
) -> np.ndarray:
    face_mask = camera_visible_face_mask(vertices, faces, dot_threshold=dot_threshold)
    mask = np.zeros((len(vertices),), dtype=bool)
    if np.any(face_mask):
        mask[np.unique(faces.astype(np.int64)[face_mask].reshape(-1))] = True
    return mask


def median_anchor(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.full((3,), np.nan, dtype=np.float32)
    return np.median(points.astype(np.float32), axis=0).astype(np.float32)


def visible_surface_anchor(
    vertices: np.ndarray,
    faces: np.ndarray,
    dot_threshold: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mask = visible_vertex_mask(vertices, faces, dot_threshold=dot_threshold)
    if not np.any(mask):
        return median_anchor(vertices), {
            "visible_vertex_count": 0,
            "total_vertex_count": int(len(vertices)),
            "visible_ratio": 0.0,
            "fallback": "whole_mesh_median",
        }
    return median_anchor(vertices[mask]), {
        "visible_vertex_count": int(mask.sum()),
        "total_vertex_count": int(len(vertices)),
        "visible_ratio": float(mask.sum() / max(1, len(vertices))),
        "fallback": None,
    }


def translate_to_keep_visible_surface_with_center_offset(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_surface_anchor: Iterable[float],
    desired_center_offset: Iterable[float],
    dot_threshold: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Place mesh so its internal center sits behind a visible surface anchor.

    ``target_surface_anchor`` is the observed depth surface anchor. The desired
    center offset is usually estimated from the mesh itself, e.g.
    ``mesh_center - visible_surface_anchor``.
    """
    visible_anchor, stats = visible_surface_anchor(vertices, faces, dot_threshold=dot_threshold)
    mesh_center = median_anchor(vertices)
    target_surface = np.asarray(list(target_surface_anchor), dtype=np.float32).reshape(3)
    desired_offset = np.asarray(list(desired_center_offset), dtype=np.float32).reshape(3)
    target_center = target_surface + desired_offset
    translation = target_center - mesh_center
    return (vertices.astype(np.float32) + translation.reshape(1, 3)).astype(np.float32), {
        "visible_anchor": visible_anchor.astype(float).tolist(),
        "mesh_center": mesh_center.astype(float).tolist(),
        "target_surface_anchor": target_surface.astype(float).tolist(),
        "desired_center_offset": desired_offset.astype(float).tolist(),
        "target_center": target_center.astype(float).tolist(),
        "translation": translation.astype(float).tolist(),
        **stats,
    }


def normalize_vector(vector: np.ndarray, fallback: Iterable[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray(list(fallback), dtype=np.float32).reshape(3)
    return (arr / norm).astype(np.float32)


def palm_facing_score_from_joints(joints21: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """Estimate whether palm/back is facing the camera from MANO 21 joints.

    Score is abs(dot(palm_normal, camera_to_hand_axis)), so both palm-facing
    and back-facing views count as "broad surface facing camera".
    """
    joints = joints21.astype(np.float32)
    wrist = joints[0]
    index_mcp = joints[5]
    middle_mcp = joints[9]
    pinky_mcp = joints[17]
    side_dir = normalize_vector(index_mcp - pinky_mcp, fallback=[1.0, 0.0, 0.0])
    forward_dir = normalize_vector(middle_mcp - wrist, fallback=[0.0, 1.0, 0.0])
    palm_normal = normalize_vector(np.cross(side_dir, forward_dir), fallback=[0.0, 0.0, 1.0])
    camera_view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    signed = float(np.dot(palm_normal, camera_view_dir))
    score = float(np.clip(abs(signed), 0.0, 1.0))
    return score, {
        "palm_normal": palm_normal.astype(float).tolist(),
        "side_dir": side_dir.astype(float).tolist(),
        "forward_dir": forward_dir.astype(float).tolist(),
        "signed_facing": signed,
        "facing_score": score,
    }


def orientation_aware_offset_m(
    joints21: np.ndarray,
    *,
    depth_bias_m: float,
    surface_min_m: float,
    surface_max_m: float,
) -> Tuple[float, Dict[str, Any]]:
    score, info = palm_facing_score_from_joints(joints21)
    surface_offset = float(surface_min_m) + score * (float(surface_max_m) - float(surface_min_m))
    total = float(depth_bias_m) + surface_offset
    return total, {
        **info,
        "depth_bias_m": float(depth_bias_m),
        "surface_min_m": float(surface_min_m),
        "surface_max_m": float(surface_max_m),
        "surface_offset_m": float(surface_offset),
        "total_offset_m": float(total),
    }


def mixed_ray_depth_offset(
    points_camera: np.ndarray,
    offset_m: float,
    ray_blend: float,
    min_z: float = 1e-6,
) -> np.ndarray:
    """Move points farther in z while blending toward constant-pixel rays."""
    points = points_camera.astype(np.float32)
    if len(points) == 0 or abs(offset_m) <= 1e-12:
        return points.copy()
    blend = float(np.clip(ray_blend, 0.0, 1.0))
    z = points[:, 2]
    z_new = z + np.float32(offset_m)
    scale = np.ones_like(z, dtype=np.float32)
    valid = np.abs(z) > min_z
    scale[valid] = z_new[valid] / z[valid]
    ray_xy = points[:, :2] * scale[:, None]
    out = points.copy()
    out[:, :2] = points[:, :2] * (1.0 - blend) + ray_xy * blend
    out[:, 2] = z_new
    return out.astype(np.float32)


def orientation_aware_compensate_points(
    points_camera: np.ndarray,
    joints21_camera: np.ndarray,
    *,
    depth_bias_m: float = 0.035,
    surface_min_m: float = 0.005,
    surface_max_m: float = 0.020,
    ray_blend: float = 0.6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    offset, info = orientation_aware_offset_m(
        joints21_camera,
        depth_bias_m=depth_bias_m,
        surface_min_m=surface_min_m,
        surface_max_m=surface_max_m,
    )
    compensated = mixed_ray_depth_offset(points_camera, offset_m=offset, ray_blend=ray_blend)
    return compensated, {**info, "ray_blend": float(ray_blend)}
