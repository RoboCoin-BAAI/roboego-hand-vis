import numpy as np
import pytest

from roboegopipe.egohand.depth_mesh import (
    crop_depth_to_pointcloud,
    core_mask_from_mask,
    depth_mask_to_pointcloud,
    filter_points_by_depth_band,
    fit_mesh_to_points_similarity,
    fit_mesh_to_points_translation,
    fit_mesh_to_pixel_depth_anchor,
    fit_mesh_z_to_points,
    hamer_rendered_obj_to_camera_opencv,
    hamer_renderer_obj_to_image,
    hamer_overlay_path_for_frame,
    lift_pixels_to_depth_points,
    mask_from_overlay_difference,
    mask_overlap_stats,
    mask_upper_region,
    mask_from_projected_points,
    mask_from_projected_triangles,
    nearest_neighbor_error,
    opencv_to_rerun_display,
    parse_obj_mesh,
    project_points_to_image,
    rotate_points_axis_angle,
    smooth_translation,
    translate_points,
)


def test_crop_depth_to_pointcloud_uses_bbox_and_intrinsics():
    depth = np.ones((4, 4), dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 10
    intrinsics = {"fx": 2.0, "fy": 2.0, "cx": 1.5, "cy": 1.5}

    points, colors, stats = crop_depth_to_pointcloud(
        depth,
        rgb,
        bbox_xyxy=[1, 1, 3, 3],
        intrinsics=intrinsics,
        stride=1,
        min_depth_m=0.2,
        max_depth_m=2.0,
        margin_px=0,
        max_points=0,
    )

    assert len(points) == 4
    assert stats["roi_pixel_count"] == 4
    assert stats["valid_depth_ratio"] == 1.0
    assert colors.shape == (4, 3)
    assert [-0.25, -0.25, 1.0] in points.tolist()


def test_mask_from_projected_points_fills_convex_hull_with_padding():
    pixels = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]], dtype=np.float32)

    mask = mask_from_projected_points(pixels, width=6, height=6, padding_px=0)

    assert mask.dtype == np.bool_
    assert mask[2, 2]
    assert not mask[0, 0]


def test_mask_from_projected_triangles_rasterizes_mesh_faces():
    pixels = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    mask = mask_from_projected_triangles(pixels, faces, width=6, height=6, padding_px=0)

    assert mask[2, 2]
    assert not mask[5, 5]


def test_mask_from_overlay_difference_keeps_component_near_bbox():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay = rgb.copy()
    overlay[2:4, 2:4] = 50
    overlay[6:8, 6:8] = 50

    mask = mask_from_overlay_difference(
        rgb,
        overlay,
        bbox_xyxy=[1.5, 1.5, 4.5, 4.5],
        diff_threshold=10.0,
        search_margin_px=1,
        padding_px=0,
    )

    assert mask[2, 2]
    assert not mask[6, 6]


def test_core_mask_from_mask_keeps_center_and_removes_edges():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True

    core = core_mask_from_mask(mask, keep_x_ratio=0.5, keep_y_ratio=0.5, min_pixels=4)

    assert core[4, 4]
    assert core[5, 5]
    assert not core[2, 2]
    assert not core[7, 7]
    assert 0 < core.sum() < mask.sum()


def test_mask_upper_region_keeps_finger_side_and_removes_wrist_side():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True

    upper = mask_upper_region(mask, keep_top_ratio=0.5, min_pixels=4)

    assert upper[2, 4]
    assert upper[4, 4]
    assert not upper[6, 4]
    assert not upper[7, 4]
    assert 0 < upper.sum() < mask.sum()


def test_mask_overlap_stats_reports_iou_and_centroid_offset():
    a = np.zeros((6, 6), dtype=bool)
    b = np.zeros((6, 6), dtype=bool)
    a[1:4, 1:4] = True
    b[2:5, 2:5] = True

    stats = mask_overlap_stats(a, b)

    assert stats["intersection_count"] == 4
    assert stats["union_count"] == 14
    assert stats["iou"] == pytest.approx(4 / 14)
    assert stats["centroid_a_xy"] == pytest.approx([2.0, 2.0])
    assert stats["centroid_b_xy"] == pytest.approx([3.0, 3.0])
    assert stats["centroid_offset_px"] == pytest.approx(2**0.5)


def test_depth_mask_to_pointcloud_uses_only_masked_valid_pixels():
    depth = np.ones((4, 4), dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 12
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    intrinsics = {"fx": 2.0, "fy": 2.0, "cx": 1.5, "cy": 1.5}

    points, colors, stats = depth_mask_to_pointcloud(
        depth,
        rgb,
        mask,
        intrinsics,
        stride=1,
        min_depth_m=0.2,
        max_depth_m=2.0,
        max_points=0,
    )

    assert len(points) == 2
    assert stats["mask_pixel_count"] == 2
    assert stats["valid_depth_ratio"] == 1.0
    assert colors[:, 0].tolist() == [12, 12]


def test_opencv_to_rerun_display_maps_rdf_camera_to_z_up_view():
    points = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, 0.5],
        ],
        dtype=np.float32,
    )

    converted = opencv_to_rerun_display(points)

    np.testing.assert_allclose(
        converted,
        np.array(
            [
                [3.0, -1.0, -2.0],
                [0.5, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
    )


def test_hamer_rendered_obj_to_camera_opencv_undoes_renderer_x_flip():
    obj_vertices = np.array(
        [
            [1.0, -2.0, -3.0],
            [-4.0, 5.0, -6.0],
        ],
        dtype=np.float32,
    )

    camera_vertices = hamer_rendered_obj_to_camera_opencv(obj_vertices)

    np.testing.assert_allclose(
        camera_vertices,
        np.array(
            [
                [1.0, 2.0, 3.0],
                [-4.0, -5.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )


def test_hamer_renderer_obj_to_image_projects_negative_z_renderer_coordinates():
    vertices = np.array(
        [
            [0.0, 0.0, -10.0],
            [1.0, -1.0, -10.0],
        ],
        dtype=np.float32,
    )

    pixels, valid = hamer_renderer_obj_to_image(vertices, focal_length=100.0, width=100, height=80)

    np.testing.assert_allclose(pixels, np.array([[50.0, 40.0], [60.0, 50.0]], dtype=np.float32))
    assert valid.tolist() == [True, True]


def test_project_points_to_image_uses_pinhole_intrinsics_and_filters_behind_camera():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.5, -0.5, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 60.0}

    pixels, valid = project_points_to_image(points, intrinsics, width=120, height=120)

    np.testing.assert_allclose(
        pixels,
        np.array([[50.0, 60.0], [100.0, 10.0]], dtype=np.float32),
        atol=1e-6,
    )
    assert valid.tolist() == [True, True, False]


def test_lift_pixels_to_depth_points_uses_depth_and_intrinsics():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    pixels = np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0]], dtype=np.float32)
    intrinsics = {"fx": 2.0, "fy": 2.0, "cx": 1.5, "cy": 1.5}

    points, valid, stats = lift_pixels_to_depth_points(
        pixels,
        depth,
        intrinsics,
        min_depth_m=0.2,
        max_depth_m=3.0,
        window_radius_px=0,
        fallback_depth_m=1.0,
    )

    np.testing.assert_allclose(
        points,
        np.array(
            [
                [-0.5, -0.5, 2.0],
                [0.5, 0.5, 2.0],
                [4.25, 4.25, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert valid.tolist() == [True, True, False]
    assert stats["depth_sample_count"] == 2
    assert stats["fallback_count"] == 1


def test_lift_pixels_to_depth_points_rejects_depth_outliers_before_backprojection():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth[1, 1] = 4.0
    pixels = np.array([[1.0, 1.0]], dtype=np.float32)
    intrinsics = {"fx": 2.0, "fy": 2.0, "cx": 1.5, "cy": 1.5}

    points, valid, stats = lift_pixels_to_depth_points(
        pixels,
        depth,
        intrinsics,
        min_depth_m=0.2,
        max_depth_m=5.0,
        window_radius_px=0,
        fallback_depth_m=2.0,
        reference_depth_m=2.0,
        max_depth_delta_m=0.3,
    )

    np.testing.assert_allclose(points, np.array([[-0.5, -0.5, 2.0]], dtype=np.float32))
    assert valid.tolist() == [False]
    assert stats["outlier_count"] == 1
    assert stats["fallback_count"] == 1


def test_hamer_overlay_path_for_frame_uses_candidates_output_layout(tmp_path):
    candidates_json = tmp_path / "ego_hand_candidates.json"
    candidates_json.write_text("{}", encoding="utf-8")

    path = hamer_overlay_path_for_frame(candidates_json, frame_idx=12, hamer_overlay_dir=None)

    assert path == tmp_path / "hamer_overlay" / "frame_000012_hamer_overlay.jpg"


def test_rotate_points_axis_angle_rotates_about_display_y_axis():
    points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    rotated = rotate_points_axis_angle(points, axis="y", degrees=90.0)

    np.testing.assert_allclose(rotated, np.array([[0.0, 0.0, -1.0]], dtype=np.float32), atol=1e-6)


def test_rotate_points_axis_angle_can_rotate_about_centroid():
    points = np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)

    rotated = rotate_points_axis_angle(points, axis="y", degrees=180.0, center=[3.0, 0.0, 0.0])

    np.testing.assert_allclose(rotated, np.array([[4.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32), atol=1e-6)


def test_translate_points_adds_xyz_offset():
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    translated = translate_points(points, [0.5, -1.0, 2.0])

    np.testing.assert_allclose(translated, np.array([[1.5, 1.0, 5.0]], dtype=np.float32))


def test_smooth_translation_uses_previous_track_with_alpha():
    previous = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    current = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    smoothed = smooth_translation(previous, current, alpha=0.25)

    np.testing.assert_allclose(smoothed, np.array([0.25, 0.5, 0.75], dtype=np.float32))


def test_smooth_translation_without_previous_returns_current():
    current = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    smoothed = smooth_translation(None, current, alpha=0.25)

    np.testing.assert_allclose(smoothed, current)


def test_fit_mesh_to_points_similarity_matches_centroid_and_extent():
    mesh = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    target = mesh * 0.5 + np.array([2.0, 3.0, 4.0], dtype=np.float32)

    fitted, transform = fit_mesh_to_points_similarity(mesh, target)

    assert fitted.mean(axis=0) == pytest.approx(target.mean(axis=0))
    assert transform["scale"] == pytest.approx(0.5)
    assert transform["translation"] == pytest.approx([2.0, 3.0, 4.0])


def test_fit_mesh_to_points_translation_keeps_original_mesh_size():
    mesh = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [9.0, 2.0, 3.0],
            [11.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )

    fitted, transform = fit_mesh_to_points_translation(mesh, target)

    assert fitted.mean(axis=0) == pytest.approx(target.mean(axis=0))
    assert np.linalg.norm(fitted[1] - fitted[0]) == pytest.approx(2.0)
    assert transform["scale"] == 1.0


def test_fit_mesh_z_to_points_only_changes_depth_axis():
    mesh = np.array(
        [
            [1.0, 2.0, 0.4],
            [3.0, 4.0, 0.6],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [100.0, 200.0, 1.4],
            [300.0, 400.0, 1.6],
        ],
        dtype=np.float32,
    )

    fitted, transform = fit_mesh_z_to_points(mesh, target)

    np.testing.assert_allclose(fitted[:, :2], mesh[:, :2])
    np.testing.assert_allclose(fitted[:, 2], np.array([1.4, 1.6], dtype=np.float32))
    assert transform["translation"] == pytest.approx([0.0, 0.0, 1.0])


def test_fit_mesh_to_pixel_depth_anchor_preserves_shape_and_uses_depth_position():
    mesh = np.array(
        [
            [-0.1, 0.0, 0.4],
            [0.1, 0.0, 0.4],
            [0.0, 0.2, 0.6],
        ],
        dtype=np.float32,
    )
    pixels = np.array(
        [
            [58.0, 46.0],
            [62.0, 44.0],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    target_points = np.array(
        [
            [1.0, 2.0, 1.8],
            [1.2, 2.2, 2.2],
        ],
        dtype=np.float32,
    )
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}

    fitted, transform = fit_mesh_to_pixel_depth_anchor(
        mesh,
        pixels,
        target_points,
        intrinsics,
    )

    np.testing.assert_allclose(fitted[1] - fitted[0], mesh[1] - mesh[0], atol=1e-6)
    np.testing.assert_allclose(fitted[2] - fitted[0], mesh[2] - mesh[0], atol=1e-6)
    assert transform["target_anchor"] == pytest.approx([0.2, 0.1, 2.0])
    np.testing.assert_allclose(np.median(fitted[:2], axis=0), np.array([0.2, 0.1, 2.0], dtype=np.float32))


def test_fit_mesh_to_pixel_depth_anchor_can_override_depth_z():
    mesh = np.array(
        [
            [-0.1, 0.0, 0.4],
            [0.1, 0.0, 0.4],
        ],
        dtype=np.float32,
    )
    pixels = np.array([[58.0, 46.0], [62.0, 44.0]], dtype=np.float32)
    target_points = np.array([[1.0, 2.0, 9.0]], dtype=np.float32)
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}

    fitted, transform = fit_mesh_to_pixel_depth_anchor(
        mesh,
        pixels,
        target_points,
        intrinsics,
        target_z_override_m=2.0,
    )

    np.testing.assert_allclose(fitted[1] - fitted[0], mesh[1] - mesh[0], atol=1e-6)
    assert transform["target_z_median"] == pytest.approx(2.0)
    assert transform["target_z_observed_median"] == pytest.approx(9.0)
    assert transform["target_z_source"] == "override"
    assert transform["target_anchor"] == pytest.approx([0.2, 0.1, 2.0])


def test_filter_points_by_depth_band_removes_background_depths():
    points = np.array(
        [
            [0.0, 0.0, 0.45],
            [0.0, 0.0, 0.50],
            [0.0, 0.0, 1.20],
        ],
        dtype=np.float32,
    )
    colors = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.uint8)

    filtered_points, filtered_colors, stats = filter_points_by_depth_band(
        points,
        colors,
        center_depth_m=0.48,
        half_width_m=0.08,
    )

    assert filtered_points[:, 2].tolist() == pytest.approx([0.45, 0.50])
    assert filtered_colors[:, 0].tolist() == [1, 2]
    assert stats["kept_ratio"] == pytest.approx(2 / 3)


def test_nearest_neighbor_error_reports_median_and_p90():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    target = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float32)

    stats = nearest_neighbor_error(source, target)

    assert stats["count"] == 2
    assert stats["median_m"] == pytest.approx(0.1)
    assert stats["p90_m"] == pytest.approx(0.18)


def test_parse_obj_mesh_reads_vertices_colors_and_faces(tmp_path):
    obj = tmp_path / "hand.obj"
    obj.write_text(
        "\n".join(
            [
                "v 0 0 0 0.1 0.2 0.3",
                "v 1 0 0 0.1 0.2 0.3",
                "v 0 1 0 0.1 0.2 0.3",
                "f 1 2 3",
            ]
        ),
        encoding="ascii",
    )

    vertices, faces, colors = parse_obj_mesh(obj)

    assert vertices.shape == (3, 3)
    assert faces.tolist() == [[0, 1, 2]]
    assert colors.tolist() == [[26, 51, 76], [26, 51, 76], [26, 51, 76]]
