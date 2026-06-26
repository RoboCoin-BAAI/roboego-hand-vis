import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.depth_anchor import DepthAnchorSelection
from roboego_hand_vis.egohand.mesh_placement import place_shape_preserved_mesh


def test_place_shape_preserved_mesh_records_anchor_source_and_count():
    mesh = np.array([[-0.1, 0.0, 0.4], [0.1, 0.0, 0.4]], dtype=np.float32)
    pixels = np.array([[58.0, 46.0], [62.0, 44.0]], dtype=np.float32)
    anchor = DepthAnchorSelection(
        points=np.array([[1.0, 2.0, 1.8], [1.2, 2.2, 2.2]], dtype=np.float32),
        colors=None,
        source="finger_upper_mask_points",
    )
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}

    fitted, transform = place_shape_preserved_mesh(
        mesh_vertices_camera=mesh,
        projected_pixels=pixels,
        depth_anchor=anchor,
        intrinsics=intrinsics,
    )

    np.testing.assert_allclose(fitted[1] - fitted[0], mesh[1] - mesh[0], atol=1e-6)
    assert transform["target_depth_points_source"] == "finger_upper_mask_points"
    assert transform["target_depth_points_count"] == 2
    assert transform["target_z_median"] == pytest.approx(2.0)


def test_place_shape_preserved_mesh_applies_recovered_depth_override():
    mesh = np.array([[-0.1, 0.0, 0.4], [0.1, 0.0, 0.4]], dtype=np.float32)
    pixels = np.array([[58.0, 46.0], [62.0, 44.0]], dtype=np.float32)
    anchor = DepthAnchorSelection(
        points=np.array([[1.0, 2.0, 9.0]], dtype=np.float32),
        colors=None,
        source="finger_upper_mask_points",
    )
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}
    override = {"target_z_m": 2.0, "source": "neighbor_normal"}

    _, transform = place_shape_preserved_mesh(
        mesh_vertices_camera=mesh,
        projected_pixels=pixels,
        depth_anchor=anchor,
        intrinsics=intrinsics,
        recovered_depth_override=override,
    )

    assert transform["target_z_median"] == pytest.approx(2.0)
    assert transform["target_z_observed_median"] == pytest.approx(9.0)
    assert transform["target_z_source"] == "override"
    assert transform["recovered_depth_override"] == override
