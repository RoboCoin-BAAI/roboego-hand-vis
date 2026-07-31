from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGERTIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]


@dataclass
class ContactResult:
    contact: bool
    min_distance_m: float | None
    contact_points: list[str]


def extract_fingertips_from_joints21(joints21: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.asarray(joints21, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected joints21 shape (21, 3), got {arr.shape}")
    return {name: arr[idx] for name, idx in zip(FINGERTIP_NAMES, FINGERTIP_INDICES)}


def compute_fingertip_contact(
    fingertips: Dict[str, np.ndarray],
    object_points: np.ndarray,
    *,
    threshold_m: float,
) -> ContactResult:
    points = np.asarray(object_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected object_points shape (N, 3), got {points.shape}")
    if len(points) == 0 or not fingertips:
        return ContactResult(False, None, [])
    tree = cKDTree(points)
    threshold = float(np.asarray(threshold_m, dtype=np.float32))
    contact_points: list[str] = []
    min_distance: float | None = None
    for name, tip in fingertips.items():
        dist, _ = tree.query(np.asarray(tip, dtype=np.float32), k=1)
        dist = float(dist)
        min_distance = dist if min_distance is None else min(min_distance, dist)
        if dist <= threshold:
            contact_points.append(name)
    return ContactResult(bool(contact_points), min_distance, contact_points)


def _side_contact_row(
    *,
    side_name: str,
    joints: np.ndarray | None,
    valid: bool,
    object_points: np.ndarray,
    threshold_m: float,
) -> tuple[bool, float | None, list[str], str]:
    if joints is None or not valid:
        return False, None, [], f"{side_name}_hand_missing"
    tips = extract_fingertips_from_joints21(joints)
    result = compute_fingertip_contact(tips, object_points, threshold_m=threshold_m)
    return (
        result.contact,
        result.min_distance_m,
        [f"{side_name}.{name}" for name in result.contact_points],
        "ok",
    )


def contact_payload_from_joints_npz_and_object_points(
    *,
    object_id: str,
    hand_joints_npz: Path,
    object_points_by_frame: dict[int, np.ndarray],
    threshold_m: float,
) -> dict:
    data = np.load(hand_joints_npz)
    left = data["left_joints_21_camera_opencv"].astype(np.float32)
    right = data["right_joints_21_camera_opencv"].astype(np.float32)
    left_valid = data["left_valid"].astype(bool)
    right_valid = data["right_valid"].astype(bool)
    n = max(len(left), len(right), max(object_points_by_frame.keys(), default=-1) + 1)
    frames = []
    for frame_idx in range(n):
        points = object_points_by_frame.get(frame_idx, np.zeros((0, 3), dtype=np.float32))
        left_contact, left_min, left_points, left_status = _side_contact_row(
            side_name="left",
            joints=left[frame_idx] if frame_idx < len(left) else None,
            valid=bool(left_valid[frame_idx]) if frame_idx < len(left_valid) else False,
            object_points=points,
            threshold_m=threshold_m,
        )
        right_contact, right_min, right_points, right_status = _side_contact_row(
            side_name="right",
            joints=right[frame_idx] if frame_idx < len(right) else None,
            valid=bool(right_valid[frame_idx]) if frame_idx < len(right_valid) else False,
            object_points=points,
            threshold_m=threshold_m,
        )
        row = {
            "frame_idx": frame_idx,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "left_min_distance_m": left_min,
            "right_min_distance_m": right_min,
            "contact_points": left_points + right_points,
            "status": "ok"
            if left_status == "ok" and right_status == "ok"
            else ",".join(status for status in [left_status, right_status] if status != "ok"),
        }
        frames.append(row)
    return {
        "schema_version": "1.0",
        "object_id": object_id,
        "distance_threshold_m": threshold_m,
        "hand_joints_npz": str(hand_joints_npz),
        "frames": frames,
    }
