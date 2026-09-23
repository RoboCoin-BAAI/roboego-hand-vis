from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import cv2
import numpy as np
import rerun as rr

from roboego_hand_vis.hand_depth_alignment.depth_mesh import (
    hamer_rendered_obj_to_camera_opencv,
    hamer_overlay_path_for_frame,
    parse_obj_mesh,
    project_points_to_image,
    translate_points,
)
from roboego_hand_vis.hand_rerun_visualization.final_smoothing import smooth_final_mesh_track
from roboego_hand_vis.hand_depth_alignment.orientation import apply_orientation_compensation as apply_mano_orientation_compensation
from roboego_hand_vis.hand_rerun_visualization.rerun_logging import log_mesh3d, log_points2d, rgb_overlay_entity, side_mesh_entity
from roboego_hand_vis.hand_rerun_visualization.temporal_tracking import SIDES
from roboego_hand_vis.hand_rerun_visualization.tracking_interface import status_layer
from linshi.scripts.export_depth_pointcloud import depth_to_pointcloud, read_rgb_frame
from linshi.scripts.rerun_utils import depth_to_vis, log_pinhole, resolve_existing_artifact_path

from .track_debug import status_color_rgba


def load_depth(depth_dir: Path, frame_idx: int) -> np.ndarray:
    path = depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def apply_alignment_orientation_compensation(
    vertices_camera: np.ndarray,
    obs: Dict[str, Any],
    regressor_by_side: Dict[str, np.ndarray],
    orientation_config: Dict[str, Any],
) -> np.ndarray:
    if not orientation_config.get("enabled"):
        return vertices_camera
    args = SimpleNamespace(
        orientation_aware_compensation=True,
        orientation_depth_bias_m=float(orientation_config.get("depth_bias_m", 0.035)),
        orientation_surface_min_m=float(orientation_config.get("surface_min_m", 0.005)),
        orientation_surface_max_m=float(orientation_config.get("surface_max_m", 0.020)),
        orientation_ray_blend=float(orientation_config.get("ray_blend", 0.3)),
    )
    compensated, _ = apply_mano_orientation_compensation(
        vertices_camera,
        obs.get("side"),
        regressor_by_side,
        args,
    )
    return compensated


def fitted_mesh_from_observation(
    obs: Dict[str, Any],
    regressor_by_side: Dict[str, np.ndarray] | None = None,
    orientation_config: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = resolve_existing_artifact_path(obs["mesh_obj"])
    vertices_obj, faces, _ = parse_obj_mesh(mesh_path)
    vertices_camera = hamer_rendered_obj_to_camera_opencv(vertices_obj)
    fitted = translate_points(vertices_camera, obs["smoothed_translation"])
    if regressor_by_side is not None and orientation_config is not None:
        fitted = apply_alignment_orientation_compensation(fitted, obs, regressor_by_side, orientation_config)
    return fitted, faces


def build_final_smoothed_meshes(
    tracks: Dict[str, List[Optional[Dict[str, Any]]]],
    *,
    window: int,
    regressor_by_side: Dict[str, np.ndarray],
    orientation_config: Dict[str, Any],
) -> Dict[str, List[Optional[Dict[str, Any]]]]:
    out: Dict[str, List[Optional[Dict[str, Any]]]] = {side: [None] * len(track) for side, track in tracks.items()}
    if window <= 0:
        return out
    for side, track in tracks.items():
        offsets: List[int] = []
        vertices_by_frame: List[np.ndarray] = []
        statuses: List[str] = []
        faces_by_frame: List[np.ndarray] = []
        for offset, obs in enumerate(track):
            if obs is None:
                continue
            vertices, faces = fitted_mesh_from_observation(
                obs,
                regressor_by_side=regressor_by_side,
                orientation_config=orientation_config,
            )
            offsets.append(offset)
            vertices_by_frame.append(vertices)
            faces_by_frame.append(faces)
            statuses.append(status_layer(obs))
        smoothed = smooth_final_mesh_track(vertices_by_frame, statuses, window=window)
        for offset, faces, item in zip(offsets, faces_by_frame, smoothed):
            out[side][offset] = {
                "vertices": item["vertices"],
                "faces": faces,
                "center_smoothing": item["center_smoothing"],
                "orientation_smoothing": item["orientation_smoothing"],
            }
    return out


def draw_mesh_on_rgb(
    image_rgb: np.ndarray,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    color_rgb: List[int],
    intrinsics: Dict[str, float],
    radius: int = 1,
) -> np.ndarray:
    out = image_rgb.copy()
    pixels, _ = project_points_to_image(
        vertices_camera,
        intrinsics,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for u, v in np.round(pixels).astype(np.int32):
        cv2.circle(out, (int(u), int(v)), radius, color_bgr, thickness=-1, lineType=cv2.LINE_AA)
    return out


def draw_bbox_and_label(
    image_rgb: np.ndarray,
    obs: Dict[str, Any],
    side: str,
    color_rgb: List[int],
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in obs.get("bbox_xyxy", [0, 0, 0, 0])]
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color_bgr, 2)
    flags = ",".join(obs.get("anomaly_flags", [])) or "ok"
    label = f"{side} {obs.get('status')} {flags}"
    y = max(18, y1 - 6)
    cv2.putText(image_rgb, label[:80], (max(0, x1), y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)


def iter_rerun_offsets(args: argparse.Namespace) -> Iterator[tuple[int, int]]:
    rrd_start = args.start if args.rrd_start is None else args.rrd_start
    if rrd_start < args.start:
        raise ValueError(f"--rrd_start {rrd_start} is before processed --start {args.start}")
    first_offset = rrd_start - args.start
    if first_offset >= args.num_frames:
        raise ValueError(
            f"--rrd_start {rrd_start} is outside processed range "
            f"[{args.start}, {args.start + args.num_frames - 1}]"
        )
    rrd_count = args.num_frames - first_offset if args.rrd_num_frames <= 0 else args.rrd_num_frames
    last_offset_exclusive = min(args.num_frames, first_offset + rrd_count)
    for offset in range(first_offset, last_offset_exclusive):
        yield offset, args.start + offset


def hand_mesh_entity_paths(side: str, obs: Dict[str, Any], log_layered_meshes: bool) -> List[str]:
    paths = [side_mesh_entity(side, "current")]
    if log_layered_meshes:
        paths.append(side_mesh_entity(side, status_layer(obs)))
    return paths


def hand_rgb_overlay_layers(side: str, obs: Dict[str, Any], log_layered_meshes: bool) -> List[str]:
    layers = ["current"]
    if log_layered_meshes:
        layer = status_layer(obs)
        if layer not in layers:
            layers.append(layer)
    return layers


def final_color_rgba(side: str) -> List[int]:
    return [60, 255, 120, 220] if side == "ego_left" else [80, 190, 255, 220]


def hand_rgb_projected_points_paths(side: str, obs: Dict[str, Any], log_layered_meshes: bool) -> List[str]:
    return [
        rgb_overlay_entity(layer, side, "mesh_projected_points")
        for layer in hand_rgb_overlay_layers(side, obs, log_layered_meshes)
    ]


def log_rerun(
    tracks: Dict[str, List[Optional[Dict[str, Any]]]],
    final_tracks: Dict[str, List[Optional[Dict[str, Any]]]],
    video_path: Path,
    depth_dir: Path,
    save_rrd: Path,
    args: argparse.Namespace,
    intrinsics: Dict[str, float],
    orientation_config: Dict[str, Any],
    regressor_by_side: Dict[str, np.ndarray],
    candidates_json: Path | None = None,
    hamer_overlay_dir: Path | None = None,
) -> None:
    rr.init("Ropedia Ego Hand Tracking Smoothed", spawn=False)
    rr.save(str(save_rrd))
    rr.log("camera_opencv", rr.ViewCoordinates.RDF, static=True)
    for offset, frame_idx in iter_rerun_offsets(args):
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        depth_m = load_depth(depth_dir, frame_idx)
        rr.set_time("frame", sequence=frame_idx)
        rr.log("camera_opencv/hands", rr.Clear(recursive=True))
        rr.log("camera_opencv/camera/left", rr.Transform3D(translation=[0, 0, 0]))
        log_pinhole("camera_opencv/camera/left", intrinsics, rgb.shape[1], rgb.shape[0])
        rr.log("camera_opencv/camera/left/rgb", rr.Image(rgb))
        rr.log("camera_opencv/camera/left/depth_vis", rr.Image(depth_to_vis(depth_m, args.min_depth_m, args.max_depth_m)))
        if candidates_json is not None:
            hamer_overlay_path = hamer_overlay_path_for_frame(candidates_json, frame_idx, hamer_overlay_dir)
            if hamer_overlay_path.exists():
                hamer_overlay_bgr = cv2.imread(str(hamer_overlay_path))
                if hamer_overlay_bgr is not None:
                    rr.log(
                        "camera_opencv/camera/left/hamer_overlay",
                        rr.Image(cv2.cvtColor(hamer_overlay_bgr, cv2.COLOR_BGR2RGB)),
                    )
        rgb_mesh_overlays = {
            "current": rgb.copy(),
            "normal": rgb.copy(),
            "recovered": rgb.copy(),
            "smooth": rgb.copy(),
            "final": rgb.copy(),
        }
        if args.log_full_scene:
            points, colors = depth_to_pointcloud(
                depth_m=depth_m,
                rgb=rgb,
                intrinsics=intrinsics,
                stride=max(1, args.scene_point_stride),
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                max_points=args.max_scene_points,
            )
            rr.log("camera_opencv/depth/full_scene_points", rr.Points3D(points, colors=colors, radii=args.point_radius))
        for side in SIDES:
            obs = tracks[side][offset]
            if obs is None:
                continue
            vertices_camera, faces = fitted_mesh_from_observation(
                obs,
                regressor_by_side=regressor_by_side,
                orientation_config=orientation_config,
            )
            non_interp_anomaly = bool([f for f in obs.get("anomaly_flags", []) if f != "interpolated"])
            color = status_color_rgba(side, obs.get("status", ""), non_interp_anomaly)
            for layer in hand_rgb_overlay_layers(side, obs, args.log_layered_meshes):
                rgb_mesh_overlays[layer] = draw_mesh_on_rgb(
                    rgb_mesh_overlays[layer],
                    vertices_camera,
                    faces,
                    color_rgb=color[:3],
                    intrinsics=intrinsics,
                    radius=max(1, int(round(args.rgb_overlay_point_radius))),
                )
                draw_bbox_and_label(rgb_mesh_overlays[layer], obs, side, color_rgb=color[:3])
            for entity_path in hand_mesh_entity_paths(side, obs, args.log_layered_meshes):
                log_mesh3d(entity_path, vertices_camera, faces, color)
            projected_pixels, _ = project_points_to_image(vertices_camera, intrinsics, width=rgb.shape[1], height=rgb.shape[0])
            if len(projected_pixels) > 0:
                for entity_path in hand_rgb_projected_points_paths(side, obs, args.log_layered_meshes):
                    log_points2d(entity_path, projected_pixels, color, radius=max(1.0, float(args.rgb_overlay_point_radius)))
            final_item = final_tracks.get(side, [None] * len(tracks[side]))[offset]
            if final_item is not None:
                final_vertices = final_item["vertices"]
                final_faces = final_item["faces"]
                final_color = final_color_rgba(side)
                log_mesh3d(side_mesh_entity(side, "final"), final_vertices, final_faces, final_color)
                rgb_mesh_overlays["final"] = draw_mesh_on_rgb(
                    rgb_mesh_overlays["final"],
                    final_vertices,
                    final_faces,
                    color_rgb=final_color[:3],
                    intrinsics=intrinsics,
                    radius=max(1, int(round(args.rgb_overlay_point_radius))),
                )
                final_pixels, _ = project_points_to_image(final_vertices, intrinsics, width=rgb.shape[1], height=rgb.shape[0])
                if len(final_pixels) > 0:
                    log_points2d(
                        rgb_overlay_entity("final", side, "mesh_projected_points"),
                        final_pixels,
                        final_color,
                        radius=max(1.0, float(args.rgb_overlay_point_radius)),
                    )
        for layer, overlay in rgb_mesh_overlays.items():
            rr.log(f"camera_opencv/camera/left/rgb_mesh_overlay/{layer}", rr.Image(overlay))
