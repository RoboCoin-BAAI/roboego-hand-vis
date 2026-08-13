from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .orientation import vertices_to_hamer21_joints
from .temporal_tracking import SIDES


HAND21_BONES: List[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

HAND21_NAMES = [
    "wrist",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_tip",
    "index_1",
    "index_2",
    "index_3",
    "index_tip",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_tip",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_tip",
    "little_1",
    "little_2",
    "little_3",
    "little_tip",
]


def vertices_by_side_to_joints21(
    vertices_by_side: Dict[str, Iterable[np.ndarray | None]],
    regressor_by_side: Dict[str, np.ndarray],
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    joints: Dict[str, np.ndarray] = {}
    valid: Dict[str, np.ndarray] = {}
    for side in SIDES:
        vertices_list = list(vertices_by_side.get(side, []))
        side_joints = np.full((len(vertices_list), 21, 3), np.nan, dtype=np.float32)
        side_valid = np.zeros((len(vertices_list),), dtype=bool)
        regressor = regressor_by_side[side]
        for idx, vertices in enumerate(vertices_list):
            if vertices is None:
                continue
            arr = np.asarray(vertices, dtype=np.float32)
            if arr.shape != (778, 3) or not np.isfinite(arr).all():
                continue
            side_joints[idx] = vertices_to_hamer21_joints(arr, regressor)
            side_valid[idx] = True
        joints[side] = side_joints
        valid[side] = side_valid
    return joints, valid


def joints_to_lines(joints: np.ndarray) -> List[np.ndarray]:
    arr = np.asarray(joints, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected joints shape (21, 3), got {arr.shape}")
    return [arr[[start, end]] for start, end in HAND21_BONES]
