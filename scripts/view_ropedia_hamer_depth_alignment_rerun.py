#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboegopipe.egohand.depth_mesh import (  # noqa: E402
    core_mask_from_mask,
    crop_depth_to_pointcloud,
    depth_mask_to_pointcloud,
    filter_points_by_depth_band,
    fit_mesh_to_pixel_depth_anchor,
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
from scripts.export_depth_pointcloud import ROPEDIA_K, depth_to_pointcloud, read_rgb_frame  # noqa: E402
from scripts.rerun_utils import depth_to_vis, log_pinhole  # noqa: E402


def resolve_path(path: str, base_dir: Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = base_dir / out
    return out.resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def selected_candidates(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected = []
    for candidate in frame["candidates"]:
        if candidate.get("status") in {
            "ego_left",
            "ego_right",
            "ego_left_recovered",
            "ego_right_recovered",
        }:
            selected.append(candidate)
    selected.sort(key=lambda c: c.get("status", "").replace("_recovered", ""))
    return selected


def canonical_hand_status(status: str) -> str:
    if status == "ego_left_recovered":
        return "ego_left"
    if status == "ego_right_recovered":
        return "ego_right"
    return status


def load_depth(depth_dir: Path, frame_idx: int) -> np.ndarray:
    path = depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[idx]


def load_tracks_by_frame(path: Path | None) -> Dict[int, Dict[str, Dict[str, Any]]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tracks = data.get("tracks", {})
    params = data.get("parameters", {})
    start = int(params.get("start", 0))
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for side, side_track in tracks.items():
        for offset, obs in enumerate(side_track):
            if obs is None:
                continue
            frame_idx = int(obs.get("frame_idx", start + offset))
            out.setdefault(frame_idx, {})[side] = obs
    return out


def build_recovered_depth_overrides(path: Path | None) -> Dict[tuple[int, str], Dict[str, Any]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    normal_z_by_side: Dict[str, list[tuple[int, float]]] = {"ego_left": [], "ego_right": []}
    recovered_frames: list[tuple[int, str, float]] = []
    for frame in report.get("frames", []):
        frame_idx = int(frame["frame_idx"])
        for hand in frame.get("hands", []):
            side = hand.get("canonical_status") or canonical_hand_status(hand.get("status", ""))
            if side not in normal_z_by_side:
                continue
            transform = hand.get("shape_preserved_fit_transform") or {}
            target_z = transform.get("target_z_median")
            if target_z is None:
                continue
            if hand.get("is_recovered"):
                recovered_frames.append((frame_idx, side, float(target_z)))
            else:
                normal_z_by_side[side].append((frame_idx, float(target_z)))

    overrides: Dict[tuple[int, str], Dict[str, Any]] = {}
    for frame_idx, side, observed_z in recovered_frames:
        normal = normal_z_by_side.get(side, [])
        if not normal:
            continue
        prev_item = None
        next_item = None
        for item in normal:
            if item[0] < frame_idx:
                prev_item = item
            elif item[0] > frame_idx:
                next_item = item
                break
        if prev_item is not None and next_item is not None:
            prev_frame, prev_z = prev_item
            next_frame, next_z = next_item
            ratio = (frame_idx - prev_frame) / float(next_frame - prev_frame)
            z = prev_z * (1.0 - ratio) + next_z * ratio
            source = {
                "mode": "interpolated_between_normal_hands",
                "prev_frame_idx": int(prev_frame),
                "next_frame_idx": int(next_frame),
                "ratio": float(ratio),
            }
        elif prev_item is not None:
            prev_frame, z = prev_item
            source = {"mode": "previous_normal_hand", "prev_frame_idx": int(prev_frame)}
        else:
            next_frame, z = next_item
            source = {"mode": "next_normal_hand", "next_frame_idx": int(next_frame)}
        overrides[(frame_idx, side)] = {
            "frame_idx": int(frame_idx),
            "side": side,
            "observed_target_z_m": float(observed_z),
            "target_z_m": float(z),
            "delta_from_observed_m": float(z - observed_z),
            **source,
        }
    return overrides


def write_recovered_depth_overrides(output_dir: Path, overrides: Dict[tuple[int, str], Dict[str, Any]]) -> None:
    if not overrides:
        return
    rows = [overrides[key] for key in sorted(overrides)]
    json_path = output_dir / "recovered_depth_overrides.json"
    csv_path = output_dir / "recovered_depth_overrides.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = [
        "frame_idx",
        "side",
        "observed_target_z_m",
        "target_z_m",
        "delta_from_observed_m",
        "mode",
        "prev_frame_idx",
        "next_frame_idx",
        "ratio",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def tracked_mesh_from_observation(obs: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mesh_vertices_obj, mesh_faces, _ = parse_obj_mesh(Path(obs["mesh_obj"]))
    vertices_camera = hamer_rendered_obj_to_camera_opencv(mesh_vertices_obj)
    vertices_smoothed = translate_points(vertices_camera, obs["smoothed_translation"])
    return vertices_smoothed, mesh_faces


def display_vertices(points_camera: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    points_display = opencv_to_rerun_display(points_camera)
    rotate_center = None
    if args.display_mesh_rotate_about == "centroid":
        rotate_center = np.median(points_display, axis=0)
    points_display = rotate_points_axis_angle(
        points_display,
        axis=args.display_mesh_rotate_axis,
        degrees=args.display_mesh_rotate_degrees,
        center=rotate_center,
    )
    return translate_points(points_display, args.display_mesh_translate)


def draw_points_rgb(image: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    if len(pixels) == 0:
        return
    h, w = image.shape[:2]
    for u, v in np.round(pixels).astype(np.int32):
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(image, (int(u), int(v)), radius, color, thickness=-1, lineType=cv2.LINE_AA)


LAYER_COLORS_RGBA = {
    "normal": [255, 180, 60, 185],
    "recovered": [255, 80, 220, 190],
    "smooth": [80, 180, 255, 195],
    "current": [60, 255, 120, 210],
}

LAYER_COLORS_RGB = {
    "normal": (255, 180, 60),
    "recovered": (255, 80, 220),
    "smooth": (80, 180, 255),
    "current": (60, 255, 120),
}


def log_mesh_layer(
    *,
    side: str,
    layer: str,
    vertices_camera: np.ndarray,
    vertices_display: np.ndarray,
    faces: np.ndarray,
    rgb_layer_images: Dict[str, np.ndarray],
    width: int,
    height: int,
    point_radius: float,
) -> None:
    color_rgba = LAYER_COLORS_RGBA[layer]
    color_rgb = LAYER_COLORS_RGB[layer]
    rr.log(
        f"camera_display/hands/{side}/layers/{layer}_mesh",
        rr.Mesh3D(
            vertex_positions=vertices_display,
            triangle_indices=faces,
            albedo_factor=color_rgba,
        ),
    )
    rr.log(
        f"camera_opencv/hands/{side}/layers/{layer}_mesh",
        rr.Mesh3D(
            vertex_positions=vertices_camera,
            triangle_indices=faces,
            albedo_factor=color_rgba,
        ),
    )
    projected_pixels, _ = project_points_to_image(vertices_camera, ROPEDIA_K, width, height)
    image = rgb_layer_images[layer]
    draw_points_rgb(image, projected_pixels, color_rgb, radius=max(1, int(round(point_radius * 0.6))))


def draw_mask_overlay_rgb(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.28,
) -> None:
    if mask.size == 0 or not mask.any():
        return
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = color
    image[:] = np.where(mask[:, :, None], image * (1.0 - alpha) + overlay * alpha, image).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness=2, lineType=cv2.LINE_AA)


def clear_hand_entities(status: str) -> None:
    rr.log(f"camera_display/hands/{status}", rr.Clear(recursive=True))
    rr.log(f"camera_opencv/hands/{status}", rr.Clear(recursive=True))
    rr.log(f"camera_display/camera/left/rgb/overlay/{status}", rr.Clear(recursive=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Ropedia HaMeR ego hands against FoundationStereo hand depth crops."
    )
    parser.add_argument(
        "--candidates_json",
        default="tmp/ropedia_hamer_ego_filter_000000_000007_tuned/ego_hand_candidates.json",
    )
    parser.add_argument(
        "--tracks_json",
        default="",
        help="Optional ego_hand_tracks.json. If set, current_best_mesh uses tracked/smoothed translation.",
    )
    parser.add_argument(
        "--recovered_depth_reference_report",
        default="",
        help=(
            "Optional previous alignment report. When set, recovered hands use target_z_median "
            "interpolated from neighboring non-recovered hands of the same side."
        ),
    )
    parser.add_argument(
        "--tracks_current_best_mode",
        choices=["replace", "fill_missing", "log_only"],
        default="replace",
        help=(
            "How tracks_json affects current_best_mesh: replace uses smoothed tracks for all tracked hands; "
            "fill_missing keeps per-frame depth-aligned mesh and only uses tracks for missing hands; "
            "log_only logs tracking diagnostics without changing current_best_mesh."
        ),
    )
    parser.add_argument(
        "--depth_dir",
        default="tmp/ropedia_foundation_depth_continuous_000000_000029",
    )
    parser.add_argument("--video", default="/home/liuyou/Documents/ropedia_data/stereo_left.mp4")
    parser.add_argument("--hamer_overlay_dir", default="")
    parser.add_argument("--output_dir", default="tmp/ropedia_hamer_depth_alignment_000000_000007")
    parser.add_argument("--save_rrd", default="")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--point_stride", type=int, default=1)
    parser.add_argument("--bbox_margin_px", type=int, default=14)
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=2.5)
    parser.add_argument("--max_points_per_hand", type=int, default=12000)
    parser.add_argument("--max_mesh_vertices_for_error", type=int, default=778)
    parser.add_argument("--point_radius", type=float, default=0.004)
    parser.add_argument("--rgb_overlay_point_radius", type=float, default=2.0)
    parser.add_argument("--max_projected_mesh_points", type=int, default=778)
    parser.add_argument("--max_projected_depth_points", type=int, default=2000)
    parser.add_argument("--log_raw_mesh", action="store_true")
    parser.add_argument("--log_raw_depth_crop", action="store_true")
    parser.add_argument("--log_full_scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--clean_final_view",
        action="store_true",
        help="Only log RGB/HaMeR overlay, depth points, final_candidate, and z_only diagnostics.",
    )
    parser.add_argument(
        "--current_best_only",
        action="store_true",
        help="Only log RGB, HaMeR overlay, RGB mesh overlay, full scene, hand_pointcloud_final, and current_best_mesh.",
    )
    parser.add_argument(
        "--new_method_only",
        action="store_true",
        help="Only log the new method chain: RGB, HaMeR overlay, full scene, hand_pointcloud_final, and current_best_mesh.",
    )
    parser.add_argument("--scene_point_stride", type=int, default=4)
    parser.add_argument("--max_scene_points", type=int, default=90000)
    parser.add_argument("--display_mesh_rotate_axis", choices=["none", "x", "y", "z"], default="none")
    parser.add_argument("--display_mesh_rotate_degrees", type=float, default=0.0)
    parser.add_argument(
        "--display_mesh_rotate_about",
        choices=["camera", "centroid"],
        default="camera",
        help="camera rotates around camera_display/camera/left origin; centroid rotates only mesh orientation.",
    )
    parser.add_argument(
        "--display_mesh_translate",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Extra translation in camera_display coordinates, after rotation. Units are meters.",
    )
    parser.add_argument("--depth_band_half_width_m", type=float, default=0.16)
    parser.add_argument(
        "--mesh_mask_source",
        choices=["coarse_fit", "raw_hamer", "hamer_renderer", "hamer_overlay_diff"],
        default="coarse_fit",
        help="Which mesh coordinates generate the 2D mask used for mask-anchor depth points.",
    )
    parser.add_argument("--mesh_mask_padding_px", type=int, default=10)
    parser.add_argument("--overlay_diff_threshold", type=float, default=8.0)
    parser.add_argument("--overlay_diff_search_margin_px", type=int, default=24)
    parser.add_argument("--mask_depth_band_half_width_m", type=float, default=0.12)
    parser.add_argument(
        "--z_anchor_source",
        choices=["mask", "core_mask"],
        default="core_mask",
        help="Depth points used to set mesh_fit_z_only. core_mask ignores mask edges before taking depth.",
    )
    parser.add_argument("--core_mask_keep_x_ratio", type=float, default=0.6)
    parser.add_argument("--core_mask_keep_y_ratio", type=float, default=0.6)
    parser.add_argument("--core_mask_min_pixels", type=int, default=20)
    parser.add_argument(
        "--finger_anchor_keep_top_ratio",
        type=float,
        default=0.65,
        help="Keep this top fraction of the hand mask as a finger/palm-side depth anchor.",
    )
    parser.add_argument("--finger_anchor_min_pixels", type=int, default=20)
    parser.add_argument(
        "--depth_lift_window_radius_px",
        type=int,
        default=2,
        help="Median depth sampling radius for lifting HaMeR renderer-projected vertices into left-camera meters.",
    )
    parser.add_argument(
        "--depth_lift_max_delta_m",
        type=float,
        default=0.12,
        help="Reject lifted vertex depths farther than this from the hand mask median depth. <=0 disables.",
    )
    parser.add_argument("--hamer_base_focal_length", type=float, default=5000.0)
    parser.add_argument("--hamer_model_image_size", type=float, default=256.0)
    parser.add_argument(
        "--smooth_translation_alpha",
        type=float,
        default=0.35,
        help="Temporal smoothing alpha for per-hand global translation. 1 disables smoothing; 0 freezes after first value.",
    )
    parser.add_argument(
        "--fit_mode",
        choices=["translation", "similarity"],
        default="translation",
        help="translation keeps HaMeR hand size; similarity independently scales each hand for diagnostics.",
    )
    return parser.parse_args()


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
    save_rrd = resolve_path(args.save_rrd, original_cwd) if args.save_rrd else (
        output_dir / "ropedia_hamer_depth_alignment.rrd"
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
        log_pinhole("camera_display/camera/left", ROPEDIA_K, rgb.shape[1], rgb.shape[0])
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
            log_pinhole("camera_opencv/camera/left", ROPEDIA_K, rgb.shape[1], rgb.shape[0])
            rr.log("camera_opencv/camera/left/rgb", rr.Image(rgb))
            rr.log(
                "camera_opencv/camera/left/depth_vis",
                rr.Image(depth_to_vis(depth_m, args.min_depth_m, args.max_depth_m)),
            )

        if args.log_full_scene:
            scene_points, scene_colors = depth_to_pointcloud(
                depth_m=depth_m,
                rgb=rgb,
                intrinsics=ROPEDIA_K,
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
            mesh_path = Path(candidate["mesh_obj"])
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
                intrinsics=ROPEDIA_K,
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
                ROPEDIA_K,
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
                ROPEDIA_K,
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
                    ROPEDIA_K,
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
                intrinsics=ROPEDIA_K,
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
                    ROPEDIA_K,
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
                ROPEDIA_K,
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
            if len(lifted_projected_pixels) > 0:
                lifted_pixels_all = np.full((len(depth_lifted_vertices), 2), np.nan, dtype=np.float32)
                lifted_pixels_all_valid, lifted_pixels_all_mask = project_points_to_image(
                    depth_lifted_vertices,
                    ROPEDIA_K,
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
                intrinsics=ROPEDIA_K,
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
                intrinsics=ROPEDIA_K,
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
                ROPEDIA_K,
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
            if len(mask_projected_pixels) > 0:
                mask_fitted_pixels_all = np.full((len(mask_fitted_vertices), 2), np.nan, dtype=np.float32)
                mask_fitted_pixels, mask_fitted_valid = project_points_to_image(
                    mask_fitted_vertices,
                    ROPEDIA_K,
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

            if args.z_anchor_source == "core_mask" and len(core_hand_points) > 0:
                z_anchor_points = core_hand_points
                z_anchor_colors = core_hand_colors
                z_anchor_source = "core_mask_points"
            elif len(mask_hand_points) > 0:
                z_anchor_points = mask_hand_points
                z_anchor_colors = mask_hand_colors
                z_anchor_source = "mask_hand_points"
            else:
                z_anchor_points = hand_points
                z_anchor_colors = hand_colors
                z_anchor_source = "bbox_hand_points"
            z_only_vertices, z_only_transform = fit_mesh_z_to_points(
                mesh_vertices_camera,
                z_anchor_points,
            )
            z_only_projected_vertices = sample_points(z_only_vertices, args.max_projected_mesh_points)
            z_only_projected_pixels, _ = project_points_to_image(
                z_only_projected_vertices,
                ROPEDIA_K,
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

            finger_z_anchor_points = finger_hand_points if len(finger_hand_points) > 0 else z_anchor_points
            finger_z_anchor_colors = finger_hand_colors if len(finger_hand_points) > 0 else z_anchor_colors
            finger_z_anchor_source = (
                "finger_upper_mask_points" if len(finger_hand_points) > 0 else z_anchor_source
            )
            z_finger_vertices, z_finger_transform = fit_mesh_z_to_points(
                mesh_vertices_camera,
                finger_z_anchor_points,
            )
            recovered_depth_override = recovered_depth_overrides.get((frame_idx, canonical_status))
            shape_preserved_vertices, shape_preserved_transform = fit_mesh_to_pixel_depth_anchor(
                mesh_vertices_camera,
                hamer_lift_pixels_all,
                mask_hand_points if len(mask_hand_points) > 0 else hand_points,
                ROPEDIA_K,
                target_z_override_m=(
                    recovered_depth_override["target_z_m"]
                    if is_recovered and recovered_depth_override is not None
                    else None
                ),
            )
            if is_recovered and recovered_depth_override is not None:
                shape_preserved_transform["recovered_depth_override"] = recovered_depth_override
            track_obs = tracks_by_frame.get(frame_idx, {}).get(canonical_status)
            tracked_vertices = None
            tracked_vertices_display = None
            tracked_faces = mesh_faces
            tracking_info = None
            if track_obs is not None:
                try:
                    tracked_vertices, tracked_faces = tracked_mesh_from_observation(track_obs)
                    tracked_vertices_display = display_vertices(tracked_vertices, args)
                    tracking_info = {
                        "status": track_obs.get("status"),
                        "source": track_obs.get("source"),
                        "mesh_obj": track_obs.get("mesh_obj"),
                        "smoothed_translation": track_obs.get("smoothed_translation"),
                        "smoothed_target_anchor": track_obs.get("smoothed_target_anchor"),
                        "anomaly_flags": track_obs.get("anomaly_flags", []),
                    }
                except FileNotFoundError:
                    tracked_vertices = None
                    tracking_info = {"error": f"missing tracked mesh: {track_obs.get('mesh_obj')}"}
            z_finger_projected_vertices = sample_points(z_finger_vertices, args.max_projected_mesh_points)
            z_finger_projected_pixels, _ = project_points_to_image(
                z_finger_projected_vertices,
                ROPEDIA_K,
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
    print(f"Rerun: {save_rrd}")


if __name__ == "__main__":
    main()
