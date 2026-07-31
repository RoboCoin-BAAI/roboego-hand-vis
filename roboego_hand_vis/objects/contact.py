from __future__ import annotations

from dataclasses import dataclass
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
    if len(points) == 0 or not fingertips:
        return ContactResult(False, None, [])
    tree = cKDTree(points)
    contact_points: list[str] = []
    min_distance: float | None = None
    for name, tip in fingertips.items():
        dist, _ = tree.query(np.asarray(tip, dtype=np.float32), k=1)
        dist = round(float(dist), 8)
        min_distance = dist if min_distance is None else min(min_distance, dist)
        if dist <= threshold_m:
            contact_points.append(name)
    return ContactResult(bool(contact_points), min_distance, contact_points)
