import numpy as np

from roboego_hand_vis.egohand.depth_anchored_joints21 import (
    anchor_joints_by_projected_xy_and_depth,
    fit_joints_xy_to_bbox,
    sample_depth_anchor,
)


def test_sample_depth_anchor_uses_median_near_wrist_and_palm_points():
    depth = np.full((20, 20), np.nan, dtype=np.float32)
    depth[8:11, 8:11] = 0.40
    depth[8:11, 12:15] = 0.60
    depth[9, 9] = 9.0
    pixels = np.asarray(
        [
            [9.0, 9.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [13.0, 9.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [13.0, 9.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [13.0, 9.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [13.0, 9.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones((21,), dtype=bool)

    anchor = sample_depth_anchor(
        depth,
        pixels,
        valid,
        mode="wrist_palm",
        window_radius_px=1,
        min_depth_m=0.15,
        max_depth_m=2.0,
    )

    assert anchor.valid is True
    assert anchor.source == "wrist_palm"
    assert anchor.sample_count == 44
    assert anchor.depth_m == np.float32(0.60)


def test_anchor_joints_by_projected_xy_preserves_2d_projection_and_relative_z():
    joints = np.asarray([[0.1 * i, 0.2 * i, 10.0 + i] for i in range(21)], dtype=np.float32)
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}
    pixels = np.asarray(
        [
            [60.0 + i, 45.0 + i]
            for i in range(21)
        ],
        dtype=np.float32,
    )
    valid = np.ones((21,), dtype=bool)

    anchored, report = anchor_joints_by_projected_xy_and_depth(
        joints,
        target_depth_m=2.0,
        pixels_xy=pixels,
        valid_pixels=valid,
        intrinsics=intrinsics,
        source_joint_index=0,
    )

    assert report["valid"] is True
    np.testing.assert_allclose(anchored[:, 2], joints[:, 2] - 8.0)
    projected_u = intrinsics["fx"] * anchored[:, 0] / anchored[:, 2] + intrinsics["cx"]
    projected_v = intrinsics["fy"] * anchored[:, 1] / anchored[:, 2] + intrinsics["cy"]
    np.testing.assert_allclose(projected_u, pixels[:, 0], atol=1e-5)
    np.testing.assert_allclose(projected_v, pixels[:, 1], atol=1e-5)


def test_fit_joints_xy_to_bbox_spreads_local_shape_over_detector_box():
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 0] = np.linspace(0.2, 0.4, 21)
    joints[:, 1] = np.linspace(0.1, 0.3, 21)

    pixels, valid = fit_joints_xy_to_bbox(
        joints,
        bbox_xyxy=[100.0, 200.0, 180.0, 260.0],
        width=320,
        height=240,
    )

    assert valid.all()
    assert float(pixels[:, 0].min()) == np.float32(100.0)
    assert float(pixels[:, 0].max()) == np.float32(180.0)
    assert float(pixels[:, 1].min()) == np.float32(200.0)
    assert float(pixels[:, 1].max()) == np.float32(239.0)


def test_sample_depth_anchor_returns_invalid_when_window_has_no_depth():
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    pixels = np.zeros((21, 2), dtype=np.float32)
    valid = np.ones((21,), dtype=bool)

    anchor = sample_depth_anchor(
        depth,
        pixels,
        valid,
        mode="wrist",
        window_radius_px=2,
        min_depth_m=0.15,
        max_depth_m=2.0,
    )

    assert anchor.valid is False
    assert anchor.depth_m is None
    assert anchor.sample_count == 0
