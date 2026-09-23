#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.hand_depth_alignment.depth_mesh import (  # noqa: E402
    core_mask_from_mask,
    crop_depth_to_pointcloud,
    depth_mask_to_pointcloud,
    filter_points_by_depth_band,
    fit_mesh_to_points_similarity,
    fit_mesh_to_points_translation,
    fit_mesh_z_to_points,
    hamer_rendered_obj_to_camera_opencv,
    hamer_renderer_obj_to_image,
    hamer_overlay_path_for_frame,
    lift_pixels_to_depth_points,
    mask_from_overlay_difference,
    mask_from_projected_points,
    mask_from_projected_triangles,
    mask_upper_region,
    nearest_neighbor_error,
    opencv_to_rerun_display,
    parse_obj_mesh,
    project_points_to_image,
    rotate_points_axis_angle,
    smooth_translation,
    translate_points,
)
from roboego_hand_vis.hand_depth_alignment.depth_anchor import (  # noqa: E402
    choose_finger_anchor,
    choose_shape_preserved_depth_anchor,
    choose_z_anchor,
)
from roboego_hand_vis.hand_depth_alignment.mesh_placement import place_shape_preserved_mesh  # noqa: E402
from roboego_hand_vis.hand_depth_alignment.orientation import (  # noqa: E402
    apply_orientation_compensation,
    load_mano_regressors,
)
from roboego_hand_vis.hand_depth_alignment.alignment_geometry import (  # noqa: E402
    display_vertices,
    draw_mask_overlay_rgb,
    draw_points_rgb,
    sample_points,
)
from roboego_hand_vis.hand_depth_alignment.alignment_io import (  # noqa: E402
    build_recovered_depth_overrides,
    canonical_hand_status,
    jsonable,
    load_depth,
    load_tracks_by_frame,
    resolve_path,
    selected_candidates,
    tracked_mesh_from_observation,
    write_recovered_depth_overrides,
)
from roboego_hand_vis.hand_depth_alignment.alignment_rerun import clear_hand_entities, log_mesh_layer  # noqa: E402
from roboego_hand_vis.hand_depth_alignment.alignment_cli import parse_args  # noqa: E402
from linshi.scripts.export_depth_pointcloud import (  # noqa: E402
    ROPEDIA_K,
    depth_to_pointcloud,
    load_camera_intrinsics,
    read_rgb_frame,
)
from linshi.scripts.rerun_utils import depth_to_vis, log_pinhole, resolve_existing_artifact_path  # noqa: E402


def main() -> None:
    original_cwd = Path.cwd()
    args = parse_args()
    if args.new_method_only:
        args.current_best_only = True
        args.clean_final_view = True
        args.log_raw_mesh = False
        args.log_raw_depth_crop = False
    candidates_json = resolve_path(args.candidates_json, original_cwd)
    tracks_json = resolve_path(args.tracks_json, original_cwd) if args.tracks_json else None
    recovered_depth_reference_report = (
        resolve_path(args.recovered_depth_reference_report, original_cwd)
        if args.recovered_depth_reference_report
        else None
    )
    depth_dir = resolve_path(args.depth_dir, original_cwd)
    video_path = resolve_path(args.video, original_cwd)
    hamer_overlay_dir = resolve_path(args.hamer_overlay_dir, original_cwd) if args.hamer_overlay_dir else None
    output_dir = resolve_path(args.output_dir, original_cwd)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.intrinsics_json:
        intrinsics_path = resolve_path(args.intrinsics_json, original_cwd)
        intrinsics = load_camera_intrinsics(
            intrinsics_path,
            resize_scale=args.intrinsics_resize_scale,
            camera=args.intrinsics_camera,
        )
    else:
        intrinsics_path = None
        intrinsics = dict(ROPEDIA_K)
    mano_model_dir = resolve_path(args.mano_model_dir, original_cwd)
    regressor_by_side = load_mano_regressors(mano_model_dir) if args.orientation_aware_compensation else {}
    save_rrd = None if args.no_save_rrd else (
        resolve_path(args.save_rrd, original_cwd) if args.save_rrd else output_dir / "ropedia_hamer_depth_alignment.rrd"
    )

    with candidates_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tracks_by_frame = load_tracks_by_frame(tracks_json)
    recovered_depth_overrides = build_recovered_depth_overrides(recovered_depth_reference_report)
    write_recovered_depth_overrides(output_dir, recovered_depth_overrides)

    report: Dict[str, Any] = {
        "candidates_json": str(candidates_json),
        "tracks_json": str(tracks_json) if tracks_json else None,
        "recovered_depth_reference_report": (
            str(recovered_depth_reference_report) if recovered_depth_reference_report else None
        ),
        "depth_dir": str(depth_dir),
        "video": str(video_path),
        "hamer_overlay_dir": str(hamer_overlay_dir) if hamer_overlay_dir else None,
        "intrinsics_json": str(intrinsics_path) if intrinsics_path else None,
        "intrinsics_camera": args.intrinsics_camera,
        "intrinsics": intrinsics,
        "shape_preserved_depth_source": args.shape_preserved_depth_source,
        "orientation_aware_compensation": {
            "enabled": bool(args.orientation_aware_compensation),
            "mano_model_dir": str(mano_model_dir) if args.orientation_aware_compensation else None,
            "depth_bias_m": args.orientation_depth_bias_m,
            "surface_min_m": args.orientation_surface_min_m,
            "surface_max_m": args.orientation_surface_max_m,
            "ray_blend": args.orientation_ray_blend,
        },
        "output_dir": str(output_dir),
        "save_rrd": str(save_rrd),
        "note": (
            "HaMeR OBJ meshes are first converted back to full-frame OpenCV camera coordinates "
            "by undoing the renderer's X-axis 180 degree OBJ export flip. mesh_fit is then "
            "aligned to the local bbox depth point cloud in camera_opencv. camera_display is "
            "only a Rerun viewing coordinate transform."
        ),
        "coordinate_frames": {
            "camera_opencv": "OpenCV/RDF camera coordinates: x right, y down, z forward.",
            "camera_display": "Rerun display coordinates derived from camera_opencv for easier viewing.",
            "hamer_obj_to_camera_opencv": "HaMeR OBJ [x, y, z] -> camera_opencv [x, -y, -z].",
        },
        "frames": [],
    }

    rr.init("Ropedia HaMeR Depth Alignment", spawn=args.spawn and not bool(save_rrd))
    if save_rrd:
        rr.save(str(save_rrd))
    rr.log("camera_display", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if not args.new_method_only:
        rr.log("camera_opencv", rr.ViewCoordinates.RDF, static=True)
    previous_smoothed_target_centroid: Dict[str, np.ndarray] = {}

    for frame in data["frames"]:
        frame_idx = int(frame["frame_idx"])
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        depth_m = load_depth(depth_dir, frame_idx)
        rr.set_time("frame", sequence=frame_idx)
        if args.new_method_only:
            rr.log("camera_display/hands/ego_left", rr.Clear(recursive=True))
            rr.log("camera_display/hands/ego_right", rr.Clear(recursive=True))
            rr.log("camera_display/hands/ego_left_recovered", rr.Clear(recursive=True))
            rr.log("camera_display/hands/ego_right_recovered", rr.Clear(recursive=True))
        else:
            clear_hand_entities("ego_left")
            clear_hand_entities("ego_right")
            clear_hand_entities("ego_left_recovered")
            clear_hand_entities("ego_right_recovered")
        rr.log("camera_display/camera/left", rr.Transform3D(translation=[0, 0, 0]))
        log_pinhole("camera_display/camera/left", intrinsics, rgb.shape[1], rgb.shape[0])
        rr.log("camera_display/camera/left/rgb", rr.Image(rgb))
        hamer_overlay_path = hamer_overlay_path_for_frame(candidates_json, frame_idx, hamer_overlay_dir)
        hamer_overlay_rgb = None
        if hamer_overlay_path.exists():
            hamer_overlay_bgr = cv2.imread(str(hamer_overlay_path))
            if hamer_overlay_bgr is not None:
                hamer_overlay_rgb = cv2.cvtColor(hamer_overlay_bgr, cv2.COLOR_BGR2RGB)
                rr.log(
                    "camera_display/camera/left/hamer_overlay",
                    rr.Image(hamer_overlay_rgb),
                )
        rr.log(
            "camera_display/camera/left/depth_vis",
            rr.Image(depth_to_vis(depth_m, args.min_depth_m, args.max_depth_m)),
        )
        if not args.new_method_only:
            rr.log("camera_opencv/camera/left", rr.Transform3D(translation=[0, 0, 0]))
            log_pinhole("camera_opencv/camera/left", intrinsics, rgb.shape[1], rgb.shape[0])
            rr.log("camera_opencv/camera/left/rgb", rr.Image(rgb))
            rr.log(
                "camera_opencv/camera/left/depth_vis",
                rr.Image(depth_to_vis(depth_m, args.min_depth_m, args.max_depth_m)),
            )

        if args.log_full_scene:
            scene_points, scene_colors = depth_to_pointcloud(
                depth_m=depth_m,
                rgb=rgb,
                intrinsics=intrinsics,
                stride=max(1, args.scene_point_stride),
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                max_points=args.max_scene_points,
            )
            rr.log(
                "camera_display/depth/full_scene_points",
                rr.Points3D(
                    opencv_to_rerun_display(scene_points),
                    colors=scene_colors,
                    radii=args.point_radius,
                ),
            )
            if not args.new_method_only:
                rr.log(
                    "camera_opencv/depth/full_scene_points",
                    rr.Points3D(scene_points, colors=scene_colors, radii=args.point_radius),
                )

        frame_report: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "hands": [],
        }
        rgb_projection_debug = rgb.copy()
        rgb_mesh_overlay_debug = rgb.copy()
        rgb_layer_images = {
            "normal": rgb.copy(),
            "recovered": rgb.copy(),
            "smooth": rgb.copy(),
            "current": rgb.copy(),
        }
        logged_track_sides: set[str] = set()

        for candidate in selected_candidates(frame):
            status = candidate["status"]
            canonical_status = canonical_hand_status(status)
            logged_track_sides.add(canonical_status)
            is_recovered = status.endswith("_recovered")
            color = [0, 220, 60, 255] if canonical_status == "ego_left" else [0, 160, 255, 255]
            if is_recovered:
                color = [255, 220, 0, 255]
            mesh_rgb_color = (0, 220, 60) if canonical_status == "ego_left" else (0, 160, 255)
            if is_recovered:
                mesh_rgb_color = (255, 220, 0)
            depth_rgb_color = (255, 230, 0) if canonical_status == "ego_left" else (255, 80, 220)
            x1, y1, x2, y2 = [float(v) for v in candidate["bbox_xyxy"]]
            cv2.rectangle(
                rgb_projection_debug,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                mesh_rgb_color,
                thickness=2,
            )
            if not args.new_method_only:
                rr.log(
                    f"camera_display/camera/left/rgb/overlay/{status}/bbox",
                    rr.Boxes2D(
                        mins=[[x1, y1]],
                        sizes=[[max(0.0, x2 - x1), max(0.0, y2 - y1)]],
                        colors=[color],
                        labels=[status],
                        show_labels=True,
                    ),
                )
            mesh_path = resolve_existing_artifact_path(candidate["mesh_obj"])
            if not mesh_path.exists():
                frame_report["hands"].append(
                    {
                        "status": status,
                        "candidate_id": candidate["candidate_id"],
                        "error": f"missing mesh: {mesh_path}",
                    }
                )
                continue

            raw_hand_points, raw_hand_colors, crop_stats = crop_depth_to_pointcloud(
                depth_m=depth_m,
                rgb=rgb,
                bbox_xyxy=candidate["bbox_xyxy"],
                intrinsics=intrinsics,
                stride=max(1, args.point_stride),
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                margin_px=args.bbox_margin_px,
                max_points=args.max_points_per_hand,
            )
            hand_points, hand_colors, depth_band_stats = filter_points_by_depth_band(
                raw_hand_points,
                raw_hand_colors,
                center_depth_m=candidate.get("median_depth_m"),
                half_width_m=args.depth_band_half_width_m,
            )
            mesh_vertices_obj, mesh_faces, _ = parse_obj_mesh(mesh_path)
            mesh_vertices_camera = hamer_rendered_obj_to_camera_opencv(mesh_vertices_obj)
            hamer_depth_lift_focal_length = (
                args.hamer_base_focal_length
                / args.hamer_model_image_size
                * max(rgb.shape[1], rgb.shape[0])
            )
            hamer_lift_pixels_all = np.full((len(mesh_vertices_obj), 2), np.nan, dtype=np.float32)
            hamer_lift_pixels, hamer_lift_valid = hamer_renderer_obj_to_image(
                mesh_vertices_obj,
                focal_length=hamer_depth_lift_focal_length,
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
            hamer_lift_pixels_all[hamer_lift_valid] = hamer_lift_pixels
            if args.fit_mode == "similarity":
                fitted_vertices, fit_transform = fit_mesh_to_points_similarity(
                    mesh_vertices_camera,
                    hand_points,
                )
            else:
                fitted_vertices, fit_transform = fit_mesh_to_points_translation(
                    mesh_vertices_camera,
                    hand_points,
                    scale=1.0,
                )
            if args.fit_mode == "similarity":
                smoothed_vertices = fitted_vertices
                smoothed_transform = {
                    "enabled": False,
                    "reason": "smooth mesh is only applied to translation fitting mode.",
                }
            else:
                current_target_centroid = np.asarray(
                    fit_transform.get("target_centroid", [0.0, 0.0, 0.0]),
                    dtype=np.float32,
                )
                current_source_centroid = np.asarray(
                    fit_transform.get("source_centroid", [0.0, 0.0, 0.0]),
                    dtype=np.float32,
                )
                smoothed_target_centroid = smooth_translation(
                    previous_smoothed_target_centroid.get(canonical_status),
                    current_target_centroid,
                    alpha=args.smooth_translation_alpha,
                )
                previous_smoothed_target_centroid[canonical_status] = smoothed_target_centroid
                smoothed_translation = smoothed_target_centroid - current_source_centroid
                smoothed_vertices = translate_points(mesh_vertices_camera, smoothed_translation)
                smoothed_transform = {
                    "enabled": True,
                    "alpha": float(np.clip(args.smooth_translation_alpha, 0.0, 1.0)),
                    "smoothed_target_centroid": smoothed_target_centroid.astype(float).tolist(),
                    "raw_target_centroid": current_target_centroid.astype(float).tolist(),
                    "source_centroid": current_source_centroid.astype(float).tolist(),
                    "translation": smoothed_translation.astype(float).tolist(),
                    "raw_translation": np.asarray(
                        fit_transform.get("translation", [0.0, 0.0, 0.0]),
                        dtype=np.float32,
                    ).astype(float).tolist(),
                }
            projected_vertices = sample_points(fitted_vertices, args.max_projected_mesh_points)
            projected_pixels, projected_valid = project_points_to_image(
                projected_vertices,
                intrinsics,
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
            if len(projected_pixels) > 0:
                if not args.new_method_only:
                    rr.log(
                        f"camera_display/camera/left/rgb/overlay/{status}/mesh_projected_points",
                        rr.Points2D(
                            projected_pixels,
                            colors=[color] * len(projected_pixels),
                            radii=args.rgb_overlay_point_radius,
                        ),
                    )
                draw_points_rgb(
                    rgb_projection_debug,
                    projected_pixels,
                    mesh_rgb_color,
                    radius=max(1, int(round(args.rgb_overlay_point_radius))),
                )
            depth_projected_points = sample_points(hand_points, args.max_projected_depth_points)
            depth_projected_pixels, depth_projected_valid = project_points_to_image(
                depth_projected_points,
                intrinsics,
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
            if len(depth_projected_pixels) > 0:
                if not args.new_method_only:
                    rr.log(
                        f"camera_display/camera/left/rgb/overlay/{status}/depth_projected_points",
                        rr.Points2D(
                            depth_projected_pixels,
                            colors=[depth_rgb_color + (210,)] * len(depth_projected_pixels),
                            radii=max(1.0, args.rgb_overlay_point_radius * 0.75),
                        ),
                    )
                draw_points_rgb(
                    rgb_projection_debug,
                    depth_projected_pixels,
                    depth_rgb_color,
                    radius=max(1, int(round(args.rgb_overlay_point_radius * 0.75))),
                )

            if args.mesh_mask_source == "hamer_overlay_diff":
                hamer_render_focal_length = None
                if hamer_overlay_rgb is None:
                    mesh_mask = np.zeros(rgb.shape[:2], dtype=bool)
                    mask_source_pixels = np.empty((0, 2), dtype=np.float32)
                else:
                    mesh_mask = mask_from_overlay_difference(
                        rgb,
                        hamer_overlay_rgb,
                        bbox_xyxy=candidate["bbox_xyxy"],
                        diff_threshold=args.overlay_diff_threshold,
                        search_margin_px=args.overlay_diff_search_margin_px,
                        padding_px=args.mesh_mask_padding_px,
                    )
                    ys_mask, xs_mask = np.where(mesh_mask)
                    mask_source_pixels = (
                        np.stack([xs_mask, ys_mask], axis=1).astype(np.float32)
                        if len(xs_mask)
                        else np.empty((0, 2), dtype=np.float32)
                    )
            elif args.mesh_mask_source == "hamer_renderer":
                hamer_render_focal_length = hamer_depth_lift_focal_length
                mask_source_pixels_all = np.full((len(mesh_vertices_obj), 2), np.nan, dtype=np.float32)
                hamer_pixels, hamer_valid = hamer_renderer_obj_to_image(
                    mesh_vertices_obj,
                    focal_length=hamer_render_focal_length,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                )
                mask_source_pixels_all[hamer_valid] = hamer_pixels
                mask_source_pixels = hamer_pixels
                mesh_mask = mask_from_projected_triangles(
                    mask_source_pixels_all,
                    mesh_faces,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                    padding_px=args.mesh_mask_padding_px,
                )
            else:
                hamer_render_focal_length = None
                mask_source_vertices = (
                    mesh_vertices_camera if args.mesh_mask_source == "raw_hamer" else fitted_vertices
                )
                mask_source_pixels, _ = project_points_to_image(
                    mask_source_vertices,
                    intrinsics,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                )
                mesh_mask = mask_from_projected_points(
                    mask_source_pixels,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                    padding_px=args.mesh_mask_padding_px,
                )
            mask_hand_points_raw, mask_hand_colors_raw, mask_stats = depth_mask_to_pointcloud(
                depth_m=depth_m,
                rgb=rgb,
                mask=mesh_mask,
                intrinsics=intrinsics,
                stride=max(1, args.point_stride),
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                max_points=args.max_points_per_hand,
            )
            mask_center_depth = (
                mask_stats.get("median_depth_m")
                if mask_stats.get("valid_depth_count", 0) > 0
                else candidate.get("median_depth_m")
            )
            mask_hand_points, mask_hand_colors, mask_depth_band_stats = filter_points_by_depth_band(
                mask_hand_points_raw,
                mask_hand_colors_raw,
                center_depth_m=mask_center_depth,
                half_width_m=args.mask_depth_band_half_width_m,
            )
            depth_lifted_vertices = np.full_like(mesh_vertices_camera, np.nan, dtype=np.float32)
            if len(hamer_lift_pixels) > 0:
                lifted_valid_vertices, lifted_depth_valid, depth_lift_stats = lift_pixels_to_depth_points(
                    hamer_lift_pixels,
                    depth_m,
                    intrinsics,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_depth_m,
                    window_radius_px=args.depth_lift_window_radius_px,
                    fallback_depth_m=mask_center_depth,
                    reference_depth_m=mask_center_depth,
                    max_depth_delta_m=args.depth_lift_max_delta_m,
                )
                depth_lifted_vertices[hamer_lift_valid] = lifted_valid_vertices
                if np.isfinite(depth_lifted_vertices).all(axis=1).any():
                    fallback_vertex = np.nanmedian(depth_lifted_vertices, axis=0)
                else:
                    fallback_z = float(mask_center_depth) if mask_center_depth is not None else 0.0
                    fallback_vertex = np.array([0.0, 0.0, fallback_z], dtype=np.float32)
                missing_vertices = ~np.isfinite(depth_lifted_vertices).all(axis=1)
                depth_lifted_vertices[missing_vertices] = fallback_vertex.astype(np.float32)
                depth_lift_stats["projected_vertex_count"] = int(len(hamer_lift_pixels))
                depth_lift_stats["missing_projected_vertex_count"] = int(missing_vertices.sum())
            else:
                fallback_z = float(mask_center_depth) if mask_center_depth is not None else 0.0
                depth_lifted_vertices[:] = np.array([0.0, 0.0, fallback_z], dtype=np.float32)
                depth_lift_stats = {
                    "input_count": int(len(mesh_vertices_obj)),
                    "depth_sample_count": 0,
                    "fallback_count": int(len(mesh_vertices_obj)),
                    "valid_depth_ratio": 0.0,
                    "fallback_depth_m": fallback_z,
                    "projected_vertex_count": 0,
                    "missing_projected_vertex_count": int(len(mesh_vertices_obj)),
                }
            lifted_projected_vertices = sample_points(depth_lifted_vertices, args.max_projected_mesh_points)
            lifted_projected_pixels, _ = project_points_to_image(
                lifted_projected_vertices,
                intrinsics,
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
            if len(lifted_projected_pixels) > 0:
                lifted_pixels_all = np.full((len(depth_lifted_vertices), 2), np.nan, dtype=np.float32)
                lifted_pixels_all_valid, lifted_pixels_all_mask = project_points_to_image(
                    depth_lifted_vertices,
                    intrinsics,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                )
                lifted_pixels_all[lifted_pixels_all_mask] = lifted_pixels_all_valid
                lifted_projected_mask = mask_from_projected_triangles(
                    lifted_pixels_all,
                    mesh_faces,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                    padding_px=0,
                )
                draw_mask_overlay_rgb(
                    rgb_mesh_overlay_debug,
                    lifted_projected_mask,
                    (60, 255, 220),
                    alpha=0.30,
                )
                if not args.new_method_only:
                    rr.log(
                        f"camera_display/camera/left/rgb/overlay/{status}/mesh_depth_lifted_projected_points",
                        rr.Points2D(
                            lifted_projected_pixels,
                            colors=[[60, 255, 220, 220]] * len(lifted_projected_pixels),
                            radii=max(1.0, args.rgb_overlay_point_radius * 0.75),
                        ),
                    )
                draw_points_rgb(
                    rgb_projection_debug,
                    lifted_projected_pixels,
                    (60, 255, 220),
                    radius=max(1, int(round(args.rgb_overlay_point_radius * 0.6))),
                )
            core_mesh_mask = core_mask_from_mask(
                mesh_mask,
                keep_x_ratio=args.core_mask_keep_x_ratio,
                keep_y_ratio=args.core_mask_keep_y_ratio,
                min_pixels=args.core_mask_min_pixels,
            )
            core_hand_points_raw, core_hand_colors_raw, core_mask_stats = depth_mask_to_pointcloud(
                depth_m=depth_m,
                rgb=rgb,
                mask=core_mesh_mask,
                intrinsics=intrinsics,
                stride=max(1, args.point_stride),
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                max_points=args.max_points_per_hand,
            )
            core_center_depth = (
                core_mask_stats.get("median_depth_m")
                if core_mask_stats.get("valid_depth_count", 0) > 0
                else mask_center_depth
            )
            core_hand_points, core_hand_colors, core_depth_band_stats = filter_points_by_depth_band(
                core_hand_points_raw,
                core_hand_colors_raw,
                center_depth_m=core_center_depth,
                half_width_m=args.mask_depth_band_half_width_m,
            )
            finger_mesh_mask = mask_upper_region(
                mesh_mask,
                keep_top_ratio=args.finger_anchor_keep_top_ratio,
                min_pixels=args.finger_anchor_min_pixels,
            )
            finger_hand_points_raw, finger_hand_colors_raw, finger_mask_stats = depth_mask_to_pointcloud(
                depth_m=depth_m,
                rgb=rgb,
                mask=finger_mesh_mask,
                intrinsics=intrinsics,
                stride=max(1, args.point_stride),
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                max_points=args.max_points_per_hand,
            )
            finger_center_depth = (
                finger_mask_stats.get("median_depth_m")
                if finger_mask_stats.get("valid_depth_count", 0) > 0
                else mask_center_depth
            )
            finger_hand_points, finger_hand_colors, finger_depth_band_stats = filter_points_by_depth_band(
                finger_hand_points_raw,
                finger_hand_colors_raw,
                center_depth_m=finger_center_depth,
                half_width_m=args.mask_depth_band_half_width_m,
            )
            if args.fit_mode == "similarity":
                mask_fitted_vertices, mask_fit_transform = fit_mesh_to_points_similarity(
                    mesh_vertices_camera,
                    mask_hand_points,
                )
            else:
                mask_fitted_vertices, mask_fit_transform = fit_mesh_to_points_translation(
                    mesh_vertices_camera,
                    mask_hand_points,
                    scale=1.0,
                )
            mask_projected_vertices = sample_points(mask_fitted_vertices, args.max_projected_mesh_points)
            mask_projected_pixels, _ = project_points_to_image(
                mask_projected_vertices,
                intrinsics,
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
            if len(mask_projected_pixels) > 0:
                mask_fitted_pixels_all = np.full((len(mask_fitted_vertices), 2), np.nan, dtype=np.float32)
                mask_fitted_pixels, mask_fitted_valid = project_points_to_image(
                    mask_fitted_vertices,
                    intrinsics,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                )
                mask_fitted_pixels_all[mask_fitted_valid] = mask_fitted_pixels
                mask_anchor_projected_mask = mask_from_projected_triangles(
                    mask_fitted_pixels_all,
                    mesh_faces,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                    padding_px=0,
                )
                draw_mask_overlay_rgb(
                    rgb_mesh_overlay_debug,
                    mask_anchor_projected_mask,
                    (255, 255, 255),
                    alpha=0.24,
                )
                if not args.new_method_only:
                    rr.log(
                        f"camera_display/camera/left/rgb/overlay/{status}/mesh_mask_anchor_projected_points",
                        rr.Points2D(
                            mask_projected_pixels,
                            colors=[[255, 255, 255, 190]] * len(mask_projected_pixels),
                            radii=max(1.0, args.rgb_overlay_point_radius * 0.75),
                        ),
                    )
                draw_points_rgb(
                    rgb_projection_debug,
                    mask_projected_pixels,
                    (255, 255, 255),
                    radius=max(1, int(round(args.rgb_overlay_point_radius * 0.6))),
                )
                if not args.new_method_only:
                    rr.log(
                        f"camera_display/camera/left/rgb/overlay/{status}/final_candidate_projected_points",
                        rr.Points2D(
                            mask_projected_pixels,
                            colors=[[255, 255, 255, 220]] * len(mask_projected_pixels),
                            radii=max(1.0, args.rgb_overlay_point_radius * 0.75),
                        ),
                    )

            z_anchor = choose_z_anchor(
                z_anchor_source=args.z_anchor_source,
                core_points=core_hand_points,
                core_colors=core_hand_colors,
                mask_points=mask_hand_points,
                mask_colors=mask_hand_colors,
                bbox_points=hand_points,
                bbox_colors=hand_colors,
            )
            z_anchor_points = z_anchor.points
            z_anchor_colors = z_anchor.colors
            z_anchor_source = z_anchor.source
            z_only_vertices, z_only_transform = fit_mesh_z_to_points(
                mesh_vertices_camera,
                z_anchor_points,
            )
            z_only_projected_vertices = sample_points(z_only_vertices, args.max_projected_mesh_points)
            z_only_projected_pixels, _ = project_points_to_image(
                z_only_projected_vertices,
                intrinsics,
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
            if len(z_only_projected_pixels) > 0:
                if not args.new_method_only:
                    rr.log(
                        f"camera_display/camera/left/rgb/overlay/{status}/mesh_z_only_projected_points",
                        rr.Points2D(
                            z_only_projected_pixels,
                            colors=[[180, 90, 255, 210]] * len(z_only_projected_pixels),
                            radii=max(1.0, args.rgb_overlay_point_radius * 0.75),
                        ),
                    )
                draw_points_rgb(
                    rgb_projection_debug,
                    z_only_projected_pixels,
                    (180, 90, 255),
                    radius=max(1, int(round(args.rgb_overlay_point_radius * 0.6))),
                )

            finger_anchor = choose_finger_anchor(
                finger_points=finger_hand_points,
                finger_colors=finger_hand_colors,
                z_anchor=z_anchor,
            )
            finger_z_anchor_points = finger_anchor.points
            finger_z_anchor_colors = finger_anchor.colors
            finger_z_anchor_source = finger_anchor.source
            z_finger_vertices, z_finger_transform = fit_mesh_z_to_points(
                mesh_vertices_camera,
                finger_z_anchor_points,
            )
            shape_preserved_anchor = choose_shape_preserved_depth_anchor(
                shape_preserved_depth_source=args.shape_preserved_depth_source,
                finger_points=finger_hand_points,
                finger_colors=finger_hand_colors,
                z_anchor=z_anchor,
                mask_points=mask_hand_points,
                bbox_points=hand_points,
            )
            shape_preserved_target_points = shape_preserved_anchor.points
            shape_preserved_depth_source = shape_preserved_anchor.source
            recovered_depth_override = recovered_depth_overrides.get((frame_idx, canonical_status))
            shape_preserved_vertices, shape_preserved_transform = place_shape_preserved_mesh(
                mesh_vertices_camera=mesh_vertices_camera,
                projected_pixels=hamer_lift_pixels_all,
                depth_anchor=shape_preserved_anchor,
                intrinsics=intrinsics,
                recovered_depth_override=(
                    recovered_depth_override if is_recovered and recovered_depth_override is not None else None
                ),
            )
            shape_preserved_vertices_uncompensated = shape_preserved_vertices
            shape_preserved_vertices, orientation_compensation_info = apply_orientation_compensation(
                shape_preserved_vertices,
                canonical_status,
                regressor_by_side,
                args,
            )
            track_obs = tracks_by_frame.get(frame_idx, {}).get(canonical_status)
            tracked_vertices = None
            tracked_vertices_display = None
            tracked_faces = mesh_faces
            tracking_info = None
            if track_obs is not None:
                try:
                    tracked_vertices, tracked_faces = tracked_mesh_from_observation(track_obs)
                    tracked_vertices_uncompensated = tracked_vertices
                    tracked_vertices, tracked_orientation_info = apply_orientation_compensation(
                        tracked_vertices,
                        canonical_status,
                        regressor_by_side,
                        args,
                    )
                    tracked_vertices_display = display_vertices(tracked_vertices, args)
                    tracking_info = {
                        "status": track_obs.get("status"),
                        "source": track_obs.get("source"),
                        "mesh_obj": track_obs.get("mesh_obj"),
                        "smoothed_translation": track_obs.get("smoothed_translation"),
                        "smoothed_target_anchor": track_obs.get("smoothed_target_anchor"),
                        "anomaly_flags": track_obs.get("anomaly_flags", []),
                        "orientation_aware_compensation": tracked_orientation_info,
                        "uncompensated_median_z": float(np.median(tracked_vertices_uncompensated[:, 2])),
                    }
                except FileNotFoundError:
                    tracked_vertices = None
                    tracking_info = {"error": f"missing tracked mesh: {track_obs.get('mesh_obj')}"}
            z_finger_projected_vertices = sample_points(z_finger_vertices, args.max_projected_mesh_points)
            z_finger_projected_pixels, _ = project_points_to_image(
                z_finger_projected_vertices,
                intrinsics,
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
            if len(z_finger_projected_pixels) > 0:
                if not args.new_method_only:
                    rr.log(
                        f"camera_display/camera/left/rgb/overlay/{status}/mesh_z_finger_anchor_projected_points",
                        rr.Points2D(
                            z_finger_projected_pixels,
                            colors=[[255, 80, 80, 215]] * len(z_finger_projected_pixels),
                            radii=max(1.0, args.rgb_overlay_point_radius * 0.75),
                        ),
                    )
                draw_points_rgb(
                    rgb_projection_debug,
                    z_finger_projected_pixels,
                    (255, 80, 80),
                    radius=max(1, int(round(args.rgb_overlay_point_radius * 0.6))),
                )

            mesh_sample = sample_points(fitted_vertices, args.max_mesh_vertices_for_error)
            mesh_to_depth = nearest_neighbor_error(mesh_sample, hand_points)
            depth_to_mesh = nearest_neighbor_error(hand_points, mesh_sample)
            smooth_mesh_sample = sample_points(smoothed_vertices, args.max_mesh_vertices_for_error)
            smooth_mesh_to_depth = nearest_neighbor_error(smooth_mesh_sample, hand_points)
            smooth_depth_to_mesh = nearest_neighbor_error(hand_points, smooth_mesh_sample)
            mask_mesh_sample = sample_points(mask_fitted_vertices, args.max_mesh_vertices_for_error)
            mask_mesh_to_depth = nearest_neighbor_error(mask_mesh_sample, mask_hand_points)
            mask_depth_to_mesh = nearest_neighbor_error(mask_hand_points, mask_mesh_sample)
            z_only_mesh_sample = sample_points(z_only_vertices, args.max_mesh_vertices_for_error)
            z_only_mesh_to_depth = nearest_neighbor_error(z_only_mesh_sample, z_anchor_points)
            z_only_depth_to_mesh = nearest_neighbor_error(z_anchor_points, z_only_mesh_sample)
            z_finger_mesh_sample = sample_points(z_finger_vertices, args.max_mesh_vertices_for_error)
            z_finger_mesh_to_depth = nearest_neighbor_error(z_finger_mesh_sample, finger_z_anchor_points)
            z_finger_depth_to_mesh = nearest_neighbor_error(finger_z_anchor_points, z_finger_mesh_sample)
            shape_preserved_mesh_sample = sample_points(shape_preserved_vertices, args.max_mesh_vertices_for_error)
            shape_preserved_mesh_to_depth = nearest_neighbor_error(shape_preserved_mesh_sample, mask_hand_points)
            shape_preserved_depth_to_mesh = nearest_neighbor_error(mask_hand_points, shape_preserved_mesh_sample)
            depth_lifted_mesh_sample = sample_points(depth_lifted_vertices, args.max_mesh_vertices_for_error)
            depth_lifted_mesh_to_depth = nearest_neighbor_error(depth_lifted_mesh_sample, mask_hand_points)
            depth_lifted_depth_to_mesh = nearest_neighbor_error(mask_hand_points, depth_lifted_mesh_sample)

            entity = f"camera_display/hands/{status}"
            fitted_vertices_display = opencv_to_rerun_display(fitted_vertices)
            rotate_center = None
            if args.display_mesh_rotate_about == "centroid":
                rotate_center = np.median(fitted_vertices_display, axis=0)
            fitted_vertices_display = rotate_points_axis_angle(
                fitted_vertices_display,
                axis=args.display_mesh_rotate_axis,
                degrees=args.display_mesh_rotate_degrees,
                center=rotate_center,
            )
            fitted_vertices_display = translate_points(
                fitted_vertices_display,
                args.display_mesh_translate,
            )
            smoothed_vertices_display = opencv_to_rerun_display(smoothed_vertices)
            smoothed_rotate_center = None
            if args.display_mesh_rotate_about == "centroid":
                smoothed_rotate_center = np.median(smoothed_vertices_display, axis=0)
            smoothed_vertices_display = rotate_points_axis_angle(
                smoothed_vertices_display,
                axis=args.display_mesh_rotate_axis,
                degrees=args.display_mesh_rotate_degrees,
                center=smoothed_rotate_center,
            )
            smoothed_vertices_display = translate_points(
                smoothed_vertices_display,
                args.display_mesh_translate,
            )
            mask_fitted_vertices_display = opencv_to_rerun_display(mask_fitted_vertices)
            mask_rotate_center = None
            if args.display_mesh_rotate_about == "centroid":
                mask_rotate_center = np.median(mask_fitted_vertices_display, axis=0)
            mask_fitted_vertices_display = rotate_points_axis_angle(
                mask_fitted_vertices_display,
                axis=args.display_mesh_rotate_axis,
                degrees=args.display_mesh_rotate_degrees,
                center=mask_rotate_center,
            )
            mask_fitted_vertices_display = translate_points(
                mask_fitted_vertices_display,
                args.display_mesh_translate,
            )
            z_only_vertices_display = opencv_to_rerun_display(z_only_vertices)
            z_only_rotate_center = None
            if args.display_mesh_rotate_about == "centroid":
                z_only_rotate_center = np.median(z_only_vertices_display, axis=0)
            z_only_vertices_display = rotate_points_axis_angle(
                z_only_vertices_display,
                axis=args.display_mesh_rotate_axis,
                degrees=args.display_mesh_rotate_degrees,
                center=z_only_rotate_center,
            )
            z_only_vertices_display = translate_points(
                z_only_vertices_display,
                args.display_mesh_translate,
            )
            z_finger_vertices_display = opencv_to_rerun_display(z_finger_vertices)
            z_finger_rotate_center = None
            if args.display_mesh_rotate_about == "centroid":
                z_finger_rotate_center = np.median(z_finger_vertices_display, axis=0)
            z_finger_vertices_display = rotate_points_axis_angle(
                z_finger_vertices_display,
                axis=args.display_mesh_rotate_axis,
                degrees=args.display_mesh_rotate_degrees,
                center=z_finger_rotate_center,
            )
            z_finger_vertices_display = translate_points(
                z_finger_vertices_display,
                args.display_mesh_translate,
            )
            depth_lifted_vertices_display = opencv_to_rerun_display(depth_lifted_vertices)
            depth_lifted_rotate_center = None
            if args.display_mesh_rotate_about == "centroid":
                depth_lifted_rotate_center = np.median(depth_lifted_vertices_display, axis=0)
            depth_lifted_vertices_display = rotate_points_axis_angle(
                depth_lifted_vertices_display,
                axis=args.display_mesh_rotate_axis,
                degrees=args.display_mesh_rotate_degrees,
                center=depth_lifted_rotate_center,
            )
            depth_lifted_vertices_display = translate_points(
                depth_lifted_vertices_display,
                args.display_mesh_translate,
            )
            shape_preserved_vertices_display = opencv_to_rerun_display(shape_preserved_vertices)
            shape_preserved_rotate_center = None
            if args.display_mesh_rotate_about == "centroid":
                shape_preserved_rotate_center = np.median(shape_preserved_vertices_display, axis=0)
            shape_preserved_vertices_display = rotate_points_axis_angle(
                shape_preserved_vertices_display,
                axis=args.display_mesh_rotate_axis,
                degrees=args.display_mesh_rotate_degrees,
                center=shape_preserved_rotate_center,
            )
            shape_preserved_vertices_display = translate_points(
                shape_preserved_vertices_display,
                args.display_mesh_translate,
            )
            if tracked_vertices is not None and args.tracks_current_best_mode == "replace":
                current_best_vertices = tracked_vertices
                current_best_faces = tracked_faces
                current_best_vertices_display = tracked_vertices_display
                current_best_source = "tracking_smoothed_shape_preserved_mano"
            else:
                current_best_vertices = shape_preserved_vertices
                current_best_faces = mesh_faces
                current_best_vertices_display = shape_preserved_vertices_display
                current_best_source = "shape_preserved_depth_aligned_mano"
            observation_layer = "recovered" if is_recovered else "normal"
            current_layer = "smooth" if current_best_source == "tracking_smoothed_shape_preserved_mano" else observation_layer
            log_mesh_layer(
                side=canonical_status,
                layer=observation_layer,
                vertices_camera=shape_preserved_vertices,
                vertices_display=shape_preserved_vertices_display,
                faces=mesh_faces,
                rgb_layer_images=rgb_layer_images,
                width=rgb.shape[1],
                height=rgb.shape[0],
                point_radius=args.rgb_overlay_point_radius,
                intrinsics=intrinsics,
            )
            if current_layer == "smooth":
                log_mesh_layer(
                    side=canonical_status,
                    layer="smooth",
                    vertices_camera=current_best_vertices,
                    vertices_display=current_best_vertices_display,
                    faces=current_best_faces,
                    rgb_layer_images=rgb_layer_images,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                    point_radius=args.rgb_overlay_point_radius,
                    intrinsics=intrinsics,
                )
            log_mesh_layer(
                side=canonical_status,
                layer="current",
                vertices_camera=current_best_vertices,
                vertices_display=current_best_vertices_display,
                faces=current_best_faces,
                rgb_layer_images=rgb_layer_images,
                width=rgb.shape[1],
                height=rgb.shape[0],
                point_radius=args.rgb_overlay_point_radius,
                intrinsics=intrinsics,
            )
            if args.log_raw_depth_crop:
                rr.log(
                    f"{entity}/depth_points_raw_crop",
                    rr.Points3D(
                        opencv_to_rerun_display(raw_hand_points),
                        colors=raw_hand_colors,
                        radii=args.point_radius,
                    ),
                )
            if not args.current_best_only:
                rr.log(
                    f"{entity}/depth_points",
                    rr.Points3D(
                        opencv_to_rerun_display(hand_points),
                        colors=hand_colors,
                        radii=args.point_radius,
                    ),
                )
                rr.log(
                    f"{entity}/depth_points_mask_anchor",
                    rr.Points3D(
                        opencv_to_rerun_display(mask_hand_points),
                        colors=mask_hand_colors,
                        radii=args.point_radius,
                    ),
                )
            rr.log(
                f"{entity}/hand_pointcloud_final",
                rr.Points3D(
                    opencv_to_rerun_display(mask_hand_points),
                    colors=mask_hand_colors,
                    radii=args.point_radius,
                ),
            )
            if not args.current_best_only:
                rr.log(
                    f"{entity}/depth_points_core_anchor",
                    rr.Points3D(
                        opencv_to_rerun_display(core_hand_points),
                        colors=core_hand_colors,
                        radii=args.point_radius,
                    ),
                )
                rr.log(
                    f"{entity}/depth_points_finger_anchor",
                    rr.Points3D(
                        opencv_to_rerun_display(finger_hand_points),
                        colors=finger_hand_colors,
                        radii=args.point_radius,
                    ),
                )
                rr.log(
                    f"{entity}/depth_points_z_anchor",
                    rr.Points3D(
                        opencv_to_rerun_display(z_anchor_points),
                        colors=z_anchor_colors,
                        radii=args.point_radius,
                    ),
                )
                rr.log(
                    f"{entity}/depth_points_z_finger_anchor",
                    rr.Points3D(
                        opencv_to_rerun_display(finger_z_anchor_points),
                        colors=finger_z_anchor_colors,
                        radii=args.point_radius,
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/depth_points",
                    rr.Points3D(hand_points, colors=hand_colors, radii=args.point_radius),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/depth_points_mask_anchor",
                    rr.Points3D(mask_hand_points, colors=mask_hand_colors, radii=args.point_radius),
                )
            if not args.new_method_only:
                rr.log(
                    f"camera_opencv/hands/{status}/hand_pointcloud_final",
                    rr.Points3D(mask_hand_points, colors=mask_hand_colors, radii=args.point_radius),
                )
            if not args.current_best_only:
                rr.log(
                    f"camera_opencv/hands/{status}/depth_points_core_anchor",
                    rr.Points3D(core_hand_points, colors=core_hand_colors, radii=args.point_radius),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/depth_points_finger_anchor",
                    rr.Points3D(finger_hand_points, colors=finger_hand_colors, radii=args.point_radius),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/depth_points_z_anchor",
                    rr.Points3D(z_anchor_points, colors=z_anchor_colors, radii=args.point_radius),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/depth_points_z_finger_anchor",
                    rr.Points3D(finger_z_anchor_points, colors=finger_z_anchor_colors, radii=args.point_radius),
                )
            if not args.clean_final_view:
                rr.log(
                    f"{entity}/mesh_fit",
                    rr.Mesh3D(
                        vertex_positions=fitted_vertices_display,
                        triangle_indices=mesh_faces,
                        albedo_factor=[0, 220, 60, 180] if status == "ego_left" else [0, 160, 255, 180],
                    ),
                )
                rr.log(
                    f"{entity}/mesh_fit_smooth",
                    rr.Mesh3D(
                        vertex_positions=smoothed_vertices_display,
                        triangle_indices=mesh_faces,
                        albedo_factor=[30, 255, 120, 105] if status == "ego_left" else [80, 210, 255, 105],
                    ),
                )
            if not args.current_best_only:
                rr.log(
                    f"{entity}/mesh_fit_mask_anchor",
                    rr.Mesh3D(
                        vertex_positions=mask_fitted_vertices_display,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 255, 255, 125],
                    ),
                )
                rr.log(
                    f"{entity}/final_candidate",
                    rr.Mesh3D(
                        vertex_positions=mask_fitted_vertices_display,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 255, 255, 145],
                    ),
                )
                rr.log(
                    f"{entity}/mesh_fit_z_only",
                    rr.Mesh3D(
                        vertex_positions=z_only_vertices_display,
                        triangle_indices=mesh_faces,
                        albedo_factor=[180, 90, 255, 135],
                    ),
                )
                rr.log(
                    f"{entity}/mesh_fit_z_finger_anchor",
                    rr.Mesh3D(
                        vertex_positions=z_finger_vertices_display,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 80, 80, 145],
                    ),
                )
            if not args.new_method_only:
                rr.log(
                    f"{entity}/mesh_depth_lifted",
                    rr.Mesh3D(
                        vertex_positions=depth_lifted_vertices_display,
                        triangle_indices=mesh_faces,
                        albedo_factor=[60, 255, 220, 165],
                    ),
                )
                rr.log(
                    f"{entity}/mesh_shape_preserved_depth_aligned",
                    rr.Mesh3D(
                        vertex_positions=shape_preserved_vertices_display,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 180, 60, 165],
                    ),
                )
            rr.log(
                f"{entity}/current_best_mesh",
                rr.Mesh3D(
                    vertex_positions=current_best_vertices_display,
                    triangle_indices=current_best_faces,
                    albedo_factor=[255, 180, 60, 185],
                ),
            )
            if tracked_vertices is not None:
                rr.log(
                    f"{entity}/tracking_smoothed_current_best_mesh",
                    rr.Mesh3D(
                        vertex_positions=tracked_vertices_display,
                        triangle_indices=tracked_faces,
                        albedo_factor=[80, 180, 255, 195],
                    ),
                )
            if not args.clean_final_view:
                rr.log(
                    f"camera_opencv/hands/{status}/mesh_fit",
                    rr.Mesh3D(
                        vertex_positions=fitted_vertices,
                        triangle_indices=mesh_faces,
                        albedo_factor=[0, 220, 60, 180] if status == "ego_left" else [0, 160, 255, 180],
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/mesh_fit_smooth",
                    rr.Mesh3D(
                        vertex_positions=smoothed_vertices,
                        triangle_indices=mesh_faces,
                        albedo_factor=[30, 255, 120, 105] if status == "ego_left" else [80, 210, 255, 105],
                    ),
                )
            if not args.current_best_only:
                rr.log(
                    f"camera_opencv/hands/{status}/mesh_fit_mask_anchor",
                    rr.Mesh3D(
                        vertex_positions=mask_fitted_vertices,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 255, 255, 125],
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/final_candidate",
                    rr.Mesh3D(
                        vertex_positions=mask_fitted_vertices,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 255, 255, 145],
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/mesh_fit_z_only",
                    rr.Mesh3D(
                        vertex_positions=z_only_vertices,
                        triangle_indices=mesh_faces,
                        albedo_factor=[180, 90, 255, 135],
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/mesh_fit_z_finger_anchor",
                    rr.Mesh3D(
                        vertex_positions=z_finger_vertices,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 80, 80, 145],
                    ),
                )
            if not args.new_method_only:
                rr.log(
                    f"camera_opencv/hands/{status}/mesh_depth_lifted",
                    rr.Mesh3D(
                        vertex_positions=depth_lifted_vertices,
                        triangle_indices=mesh_faces,
                        albedo_factor=[60, 255, 220, 165],
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/mesh_shape_preserved_depth_aligned",
                    rr.Mesh3D(
                        vertex_positions=shape_preserved_vertices,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 180, 60, 165],
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/current_best_mesh",
                    rr.Mesh3D(
                        vertex_positions=current_best_vertices,
                        triangle_indices=current_best_faces,
                        albedo_factor=[255, 180, 60, 185],
                    ),
                )
                if tracked_vertices is not None:
                    rr.log(
                        f"camera_opencv/hands/{status}/tracking_smoothed_current_best_mesh",
                        rr.Mesh3D(
                            vertex_positions=tracked_vertices,
                            triangle_indices=tracked_faces,
                            albedo_factor=[80, 180, 255, 195],
                        ),
                    )
            if args.log_raw_mesh:
                rr.log(
                    f"{entity}/mesh_raw_hamer_camera_opencv",
                    rr.Mesh3D(
                        vertex_positions=opencv_to_rerun_display(mesh_vertices_camera),
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 120, 0, 120],
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/mesh_raw_hamer_camera_opencv",
                    rr.Mesh3D(
                        vertex_positions=mesh_vertices_camera,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 120, 0, 120],
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{status}/mesh_raw_hamer_obj_export_coords",
                    rr.Mesh3D(
                        vertex_positions=mesh_vertices_obj,
                        triangle_indices=mesh_faces,
                        albedo_factor=[255, 80, 0, 80],
                    ),
                )

            hand_report = {
                "status": status,
                "canonical_status": canonical_status,
                "is_recovered": is_recovered,
                "recovery": candidate.get("recovery"),
                "candidate_id": candidate["candidate_id"],
                "bbox_xyxy": candidate["bbox_xyxy"],
                "mesh_obj": str(mesh_path),
                "crop_stats": crop_stats,
                "depth_band_stats": depth_band_stats,
                "fit_transform": fit_transform,
                "fit_mode": args.fit_mode,
                "smooth_fit_transform": smoothed_transform,
                "mask_fit_transform": mask_fit_transform,
                "z_only_fit_transform": z_only_transform,
                "z_only_anchor_source": z_anchor_source,
                "z_anchor_source_arg": args.z_anchor_source,
                "z_finger_fit_transform": z_finger_transform,
                "z_finger_anchor_source": finger_z_anchor_source,
                "shape_preserved_fit_transform": shape_preserved_transform,
                "shape_preserved_depth_source": shape_preserved_depth_source,
                "depth_anchor": {
                    "z_anchor": z_anchor.report(),
                    "finger_anchor": finger_anchor.report(),
                    "shape_preserved_anchor": shape_preserved_anchor.report(),
                    "shape_preserved_depth_source_arg": args.shape_preserved_depth_source,
                },
                "orientation_aware_compensation": orientation_compensation_info,
                "shape_preserved_uncompensated_median_z": float(
                    np.median(shape_preserved_vertices_uncompensated[:, 2])
                ),
                "shape_preserved_compensated_median_z": float(np.median(shape_preserved_vertices[:, 2])),
                "current_best_mesh_source": current_best_source,
                "layer": observation_layer,
                "current_layer": current_layer,
                "tracking_smoothed": tracking_info,
                "depth_lift_stats": depth_lift_stats,
                "mesh_mask_source": args.mesh_mask_source,
                "hamer_render_focal_length": hamer_render_focal_length,
                "hamer_depth_lift_focal_length": hamer_depth_lift_focal_length,
                "mesh_coordinate_source": "hamer_obj_converted_to_camera_opencv",
                "hamer_obj_to_camera_opencv": "[x, y, z] -> [x, -y, -z]",
                "rgb_projection": {
                    "projected_count": int(len(projected_pixels)),
                    "input_count": int(len(projected_vertices)),
                    "visible_ratio": float(len(projected_pixels) / max(1, len(projected_vertices))),
                },
                "depth_rgb_projection": {
                    "projected_count": int(len(depth_projected_pixels)),
                    "input_count": int(len(depth_projected_points)),
                    "visible_ratio": float(len(depth_projected_pixels) / max(1, len(depth_projected_points))),
                },
                "display_mesh_rotation": {
                    "axis": args.display_mesh_rotate_axis,
                    "degrees": args.display_mesh_rotate_degrees,
                    "about": args.display_mesh_rotate_about,
                },
                "display_mesh_translate": args.display_mesh_translate,
                "mesh_to_depth_error": mesh_to_depth,
                "depth_to_mesh_error": depth_to_mesh,
                "smooth_mesh_to_depth_error": smooth_mesh_to_depth,
                "smooth_depth_to_mesh_error": smooth_depth_to_mesh,
                "mask_stats": mask_stats,
                "mask_depth_band_stats": mask_depth_band_stats,
                "core_mask_stats": core_mask_stats,
                "core_depth_band_stats": core_depth_band_stats,
                "finger_mask_stats": finger_mask_stats,
                "finger_depth_band_stats": finger_depth_band_stats,
                "mask_mesh_to_depth_error": mask_mesh_to_depth,
                "mask_depth_to_mesh_error": mask_depth_to_mesh,
                "z_only_mesh_to_depth_error": z_only_mesh_to_depth,
                "z_only_depth_to_mesh_error": z_only_depth_to_mesh,
                "z_finger_mesh_to_depth_error": z_finger_mesh_to_depth,
                "z_finger_depth_to_mesh_error": z_finger_depth_to_mesh,
                "shape_preserved_mesh_to_depth_error": shape_preserved_mesh_to_depth,
                "shape_preserved_depth_to_mesh_error": shape_preserved_depth_to_mesh,
                "depth_lifted_mesh_to_depth_error": depth_lifted_mesh_to_depth,
                "depth_lifted_depth_to_mesh_error": depth_lifted_depth_to_mesh,
            }
            frame_report["hands"].append(hand_report)
            print(
                f"frame={frame_idx:06d} {status} "
                f"depth_points={len(hand_points)}/{len(raw_hand_points)} "
                f"band_kept={depth_band_stats.get('kept_ratio', 0.0):.3f} "
                f"mesh2depth_med={mesh_to_depth['median_m']} "
                f"depth2mesh_med={depth_to_mesh['median_m']} "
                f"smooth_mesh2depth_med={smooth_mesh_to_depth['median_m']} "
                f"mask_mesh2depth_med={mask_mesh_to_depth['median_m']} "
                f"zonly_mesh2depth_med={z_only_mesh_to_depth['median_m']} "
                f"zfinger_mesh2depth_med={z_finger_mesh_to_depth['median_m']} "
                f"shape_preserved_mesh2depth_med={shape_preserved_mesh_to_depth['median_m']} "
                f"lifted_mesh2depth_med={depth_lifted_mesh_to_depth['median_m']}"
            )

        for side, track_obs in tracks_by_frame.get(frame_idx, {}).items():
            if side in logged_track_sides:
                continue
            if args.tracks_current_best_mode == "log_only":
                continue
            try:
                tracked_vertices, tracked_faces = tracked_mesh_from_observation(track_obs)
            except FileNotFoundError:
                frame_report["hands"].append(
                    {
                        "status": side,
                        "canonical_status": side,
                        "current_best_mesh_source": "tracking_smoothed_shape_preserved_mano",
                        "tracking_smoothed": {
                            "error": f"missing tracked mesh: {track_obs.get('mesh_obj')}",
                            "status": track_obs.get("status"),
                            "source": track_obs.get("source"),
                        },
                    }
                )
                continue
            tracked_vertices_uncompensated = tracked_vertices
            tracked_vertices, tracked_orientation_info = apply_orientation_compensation(
                tracked_vertices,
                side,
                regressor_by_side,
                args,
            )
            tracked_vertices_display = display_vertices(tracked_vertices, args)
            entity = f"camera_display/hands/{side}"
            log_mesh_layer(
                side=side,
                layer="smooth",
                vertices_camera=tracked_vertices,
                vertices_display=tracked_vertices_display,
                faces=tracked_faces,
                rgb_layer_images=rgb_layer_images,
                width=rgb.shape[1],
                height=rgb.shape[0],
                point_radius=args.rgb_overlay_point_radius,
                intrinsics=intrinsics,
            )
            log_mesh_layer(
                side=side,
                layer="current",
                vertices_camera=tracked_vertices,
                vertices_display=tracked_vertices_display,
                faces=tracked_faces,
                rgb_layer_images=rgb_layer_images,
                width=rgb.shape[1],
                height=rgb.shape[0],
                point_radius=args.rgb_overlay_point_radius,
                intrinsics=intrinsics,
            )
            rr.log(
                f"{entity}/current_best_mesh",
                rr.Mesh3D(
                    vertex_positions=tracked_vertices_display,
                    triangle_indices=tracked_faces,
                    albedo_factor=[80, 180, 255, 195],
                ),
            )
            rr.log(
                f"{entity}/tracking_smoothed_current_best_mesh",
                rr.Mesh3D(
                    vertex_positions=tracked_vertices_display,
                    triangle_indices=tracked_faces,
                    albedo_factor=[80, 180, 255, 210],
                ),
            )
            if not args.new_method_only:
                rr.log(
                    f"camera_opencv/hands/{side}/current_best_mesh",
                    rr.Mesh3D(
                        vertex_positions=tracked_vertices,
                        triangle_indices=tracked_faces,
                        albedo_factor=[80, 180, 255, 195],
                    ),
                )
                rr.log(
                    f"camera_opencv/hands/{side}/tracking_smoothed_current_best_mesh",
                    rr.Mesh3D(
                        vertex_positions=tracked_vertices,
                        triangle_indices=tracked_faces,
                        albedo_factor=[80, 180, 255, 210],
                    ),
                )
            frame_report["hands"].append(
                {
                    "status": side,
                    "canonical_status": side,
                    "is_recovered": False,
                    "candidate_id": track_obs.get("candidate_id"),
                    "bbox_xyxy": track_obs.get("bbox_xyxy"),
                    "mesh_obj": track_obs.get("mesh_obj"),
                    "current_best_mesh_source": "tracking_smoothed_shape_preserved_mano",
                    "layer": "smooth",
                    "current_layer": "smooth",
                    "tracking_smoothed": {
                        "status": track_obs.get("status"),
                        "source": track_obs.get("source"),
                        "smoothed_translation": track_obs.get("smoothed_translation"),
                        "smoothed_target_anchor": track_obs.get("smoothed_target_anchor"),
                        "anomaly_flags": track_obs.get("anomaly_flags", []),
                        "interpolation": track_obs.get("interpolation"),
                        "orientation_aware_compensation": tracked_orientation_info,
                        "uncompensated_median_z": float(np.median(tracked_vertices_uncompensated[:, 2])),
                        "compensated_median_z": float(np.median(tracked_vertices[:, 2])),
                    },
                }
            )
            print(
                f"frame={frame_idx:06d} {side} "
                f"tracking_smoothed_status={track_obs.get('status')} "
                f"interpolated={track_obs.get('status') == 'interpolated'}"
            )

        debug_dir = output_dir / "rgb_projection_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"frame_{frame_idx:06d}_rgb_projection_debug.jpg"
        cv2.imwrite(str(debug_path), cv2.cvtColor(rgb_projection_debug, cv2.COLOR_RGB2BGR))
        if not args.new_method_only:
            rr.log("camera_display/camera/left/rgb_projection_debug", rr.Image(rgb_projection_debug))
        mesh_overlay_debug_path = debug_dir / f"frame_{frame_idx:06d}_rgb_mesh_overlay_debug.jpg"
        cv2.imwrite(str(mesh_overlay_debug_path), cv2.cvtColor(rgb_mesh_overlay_debug, cv2.COLOR_RGB2BGR))
        if not args.new_method_only:
            rr.log("camera_display/camera/left/rgb_mesh_overlay_debug", rr.Image(rgb_mesh_overlay_debug))
        for layer, layer_image in rgb_layer_images.items():
            layer_path = debug_dir / f"frame_{frame_idx:06d}_rgb_{layer}_mesh_overlay.jpg"
            cv2.imwrite(str(layer_path), cv2.cvtColor(layer_image, cv2.COLOR_RGB2BGR))
            rr.log(f"camera_display/camera/left/rgb/layers/{layer}_mesh_overlay", rr.Image(layer_image))
        report["frames"].append(frame_report)

    report_path = output_dir / "hand_depth_mesh_alignment_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, ensure_ascii=False, indent=2)

    print(f"Report: {report_path}")
    if save_rrd:
        print(f"Rerun: {save_rrd}")
    else:
        print("Rerun: disabled")


if __name__ == "__main__":
    main()
