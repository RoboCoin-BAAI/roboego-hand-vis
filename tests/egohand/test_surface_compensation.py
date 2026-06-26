import numpy as np
import pytest

from roboego_hand_vis.egohand.surface_compensation import (
    mixed_ray_depth_offset,
    orientation_aware_offset_m,
    translate_to_keep_visible_surface_with_center_offset,
    visible_surface_anchor,
    visible_vertex_mask,
)


def test_visible_vertex_mask_selects_surface_facing_camera_origin():
    # A square at z=1 with outward normal -z faces the camera at the origin.
    vertices = np.array(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.2],
            [1.0, -1.0, 1.2],
            [1.0, 1.0, 1.2],
            [-1.0, 1.0, 1.2],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
        ],
        dtype=np.int32,
    )

    mask = visible_vertex_mask(vertices, faces)
    anchor, stats = visible_surface_anchor(vertices, faces)

    assert np.where(mask)[0].tolist() == [0, 1, 2, 3]
    assert anchor == pytest.approx([0.0, 0.0, 1.0])
    assert stats["visible_vertex_count"] == 4


def test_translate_to_keep_visible_surface_places_center_behind_observed_surface():
    vertices = np.array(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.1],
            [1.0, -1.0, 1.1],
            [1.0, 1.0, 1.1],
            [-1.0, 1.0, 1.1],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]], dtype=np.int32)

    fitted, transform = translate_to_keep_visible_surface_with_center_offset(
        vertices,
        faces,
        target_surface_anchor=[0.0, 0.0, 2.0],
        desired_center_offset=[0.0, 0.0, 0.05],
    )

    assert transform["target_center"] == pytest.approx([0.0, 0.0, 2.05])
    assert np.median(fitted, axis=0) == pytest.approx([0.0, 0.0, 2.05])


def test_orientation_aware_offset_is_larger_when_palm_faces_camera():
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[0] = [0.0, 0.0, 1.0]  # wrist
    joints[5] = [1.0, 0.0, 1.0]  # index MCP
    joints[9] = [0.0, 1.0, 1.0]  # middle MCP
    joints[17] = [-1.0, 0.0, 1.0]  # pinky MCP

    offset, info = orientation_aware_offset_m(
        joints,
        depth_bias_m=0.035,
        surface_min_m=0.005,
        surface_max_m=0.020,
    )

    assert info["facing_score"] == pytest.approx(1.0)
    assert offset == pytest.approx(0.055)


def test_orientation_aware_offset_is_smaller_when_hand_is_sideways():
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[0] = [0.0, 0.0, 1.0]
    joints[5] = [1.0, 0.0, 1.0]
    joints[9] = [0.0, 0.0, 2.0]
    joints[17] = [-1.0, 0.0, 1.0]

    offset, info = orientation_aware_offset_m(
        joints,
        depth_bias_m=0.035,
        surface_min_m=0.005,
        surface_max_m=0.020,
    )

    assert info["facing_score"] == pytest.approx(0.0)
    assert offset == pytest.approx(0.040)


def test_mixed_ray_depth_offset_preserves_projection_at_full_blend():
    points = np.array([[0.1, -0.05, 0.5]], dtype=np.float32)

    moved = mixed_ray_depth_offset(points, offset_m=0.05, ray_blend=1.0)

    assert moved[0, 2] == pytest.approx(0.55)
    assert moved[0, 0] / moved[0, 2] == pytest.approx(points[0, 0] / points[0, 2])
    assert moved[0, 1] / moved[0, 2] == pytest.approx(points[0, 1] / points[0, 2])
