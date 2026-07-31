import numpy as np

from roboego_hand_vis.objects.depth_geometry import (
    MaskDepthStats,
    depth_path_for_frame,
    load_depth_frame,
    mask_depth_to_points,
)


def test_mask_depth_to_points_projects_pixels_to_camera_frame():
    depth = np.ones((3, 3), dtype=np.float32) * 2.0
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255
    points, stats = mask_depth_to_points(
        depth,
        mask,
        intrinsics={"fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 1.0},
        min_depth_m=0.1,
        max_depth_m=5.0,
        stride=1,
    )
    assert isinstance(stats, MaskDepthStats)
    assert points.tolist() == [[0.0, 0.0, 2.0]]
    assert stats.depth_median_m == 2.0
    assert stats.depth_valid_ratio == 1.0


def test_mask_depth_to_points_filters_invalid_depth():
    depth = np.array([[np.nan, 1.0], [10.0, 2.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=np.uint8) * 255
    points, stats = mask_depth_to_points(
        depth,
        mask,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        min_depth_m=0.5,
        max_depth_m=3.0,
        stride=1,
    )
    assert len(points) == 2
    assert stats.valid_depth_count == 2
    assert stats.mask_pixel_count == 4


def test_mask_depth_to_points_reports_unsampled_valid_depth_stats():
    depth = np.ones((3, 3), dtype=np.float32) * 2.0
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255
    points, stats = mask_depth_to_points(
        depth,
        mask,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        min_depth_m=0.1,
        max_depth_m=5.0,
        stride=2,
    )
    assert points.shape == (0, 3)
    assert stats.valid_depth_count == 1
    assert stats.depth_valid_ratio == 1.0
    assert stats.depth_median_m == 2.0


def test_mask_depth_to_points_stats_include_valid_unsampled_depth():
    depth = np.array([[np.nan, 0.0], [0.0, 3.0]], dtype=np.float32)
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 255
    mask[1, 1] = 255
    points, stats = mask_depth_to_points(
        depth,
        mask,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        min_depth_m=0.5,
        max_depth_m=5.0,
        stride=2,
    )
    assert points.shape == (0, 3)
    assert stats.valid_depth_count == 1
    assert stats.depth_valid_ratio == 0.5
    assert stats.depth_median_m == 3.0


def test_mask_depth_to_points_rejects_shape_mismatch():
    depth = np.ones((2, 2), dtype=np.float32)
    mask = np.ones((3, 3), dtype=np.uint8)
    try:
        mask_depth_to_points(
            depth,
            mask,
            intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            min_depth_m=0.1,
            max_depth_m=5.0,
            stride=1,
        )
    except ValueError as exc:
        assert "shape" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for mask/depth shape mismatch")


def test_load_depth_frame_uses_expected_path_and_float32(tmp_path):
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    path = depth_path_for_frame(depth_dir, 3)
    np.save(path, np.array([[1.0]], dtype=np.float64))

    loaded = load_depth_frame(depth_dir, 3)

    assert path.name == "frame_000003_foundation_depth_m.npy"
    assert loaded.dtype == np.float32
    assert loaded.tolist() == [[1.0]]
