import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.depth_anchor import (
    choose_shape_preserved_depth_anchor,
    choose_z_anchor,
)


def points(count: int, z: float) -> np.ndarray:
    arr = np.zeros((count, 3), dtype=np.float32)
    arr[:, 2] = z
    return arr


def colors(count: int) -> np.ndarray:
    return np.zeros((count, 3), dtype=np.uint8)


def test_choose_z_anchor_prefers_core_mask_when_requested_and_available():
    selected = choose_z_anchor(
        z_anchor_source="core_mask",
        core_points=points(3, 0.3),
        core_colors=colors(3),
        mask_points=points(4, 0.4),
        mask_colors=colors(4),
        bbox_points=points(5, 0.5),
        bbox_colors=colors(5),
    )

    assert selected.source == "core_mask_points"
    assert len(selected.points) == 3


def test_choose_z_anchor_falls_back_to_mask_then_bbox():
    selected = choose_z_anchor(
        z_anchor_source="core_mask",
        core_points=points(0, 0.0),
        core_colors=colors(0),
        mask_points=points(4, 0.4),
        mask_colors=colors(4),
        bbox_points=points(5, 0.5),
        bbox_colors=colors(5),
    )

    assert selected.source == "mask_hand_points"
    assert len(selected.points) == 4

    selected = choose_z_anchor(
        z_anchor_source="core_mask",
        core_points=points(0, 0.0),
        core_colors=colors(0),
        mask_points=points(0, 0.0),
        mask_colors=colors(0),
        bbox_points=points(5, 0.5),
        bbox_colors=colors(5),
    )

    assert selected.source == "bbox_hand_points"
    assert len(selected.points) == 5


def test_choose_shape_preserved_depth_anchor_prefers_finger_source():
    z_anchor = choose_z_anchor(
        z_anchor_source="core_mask",
        core_points=points(0, 0.0),
        core_colors=colors(0),
        mask_points=points(4, 0.4),
        mask_colors=colors(4),
        bbox_points=points(5, 0.5),
        bbox_colors=colors(5),
    )

    selected = choose_shape_preserved_depth_anchor(
        shape_preserved_depth_source="finger",
        finger_points=points(6, 0.2),
        finger_colors=colors(6),
        z_anchor=z_anchor,
        mask_points=points(4, 0.4),
        bbox_points=points(5, 0.5),
    )

    assert selected.source == "finger_upper_mask_points"
    assert len(selected.points) == 6


def test_choose_shape_preserved_depth_anchor_falls_back_to_z_anchor_source():
    z_anchor = choose_z_anchor(
        z_anchor_source="core_mask",
        core_points=points(0, 0.0),
        core_colors=colors(0),
        mask_points=points(4, 0.4),
        mask_colors=colors(4),
        bbox_points=points(5, 0.5),
        bbox_colors=colors(5),
    )

    selected = choose_shape_preserved_depth_anchor(
        shape_preserved_depth_source="finger",
        finger_points=points(0, 0.0),
        finger_colors=colors(0),
        z_anchor=z_anchor,
        mask_points=points(4, 0.4),
        bbox_points=points(5, 0.5),
    )

    assert selected.source == "mask_hand_points"
    assert len(selected.points) == 4

    z_anchor = choose_z_anchor(
        z_anchor_source="core_mask",
        core_points=points(0, 0.0),
        core_colors=colors(0),
        mask_points=points(0, 0.0),
        mask_colors=colors(0),
        bbox_points=points(5, 0.5),
        bbox_colors=colors(5),
    )
    selected = choose_shape_preserved_depth_anchor(
        shape_preserved_depth_source="finger",
        finger_points=points(0, 0.0),
        finger_colors=colors(0),
        z_anchor=z_anchor,
        mask_points=points(0, 0.0),
        bbox_points=points(5, 0.5),
    )

    assert selected.source == "bbox_hand_points"
    assert len(selected.points) == 5
