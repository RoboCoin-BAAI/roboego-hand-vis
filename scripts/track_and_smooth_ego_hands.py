#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboegopipe.egohand.depth_mesh import (  # noqa: E402
    hamer_rendered_obj_to_camera_opencv,
    opencv_to_rerun_display,
    parse_obj_mesh,
    project_points_to_image,
    translate_points,
)
from roboegopipe.egohand.temporal_tracking import (  # noqa: E402
    SIDES,
    process_tracks,
    track_summary,
)
from scripts.export_depth_pointcloud import ROPEDIA_K, depth_to_pointcloud, read_rgb_frame  # noqa: E402
from scripts.rerun_utils import depth_to_vis, log_pinhole  # noqa: E402


def resolve_path(path: str | Path, base_dir: Path) -> Path:
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
        return {str(k): jsonable(v) for k, v in value.items() if k != "source_hand"}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def status_color_rgba(side: str, status: str, has_anomaly: bool) -> List[int]:
    if has_anomaly:
        return [255, 60, 60, 205]
    if status == "interpolated":
        return [80, 150, 255, 180]
    if status == "recovered":
        return [255, 220, 60, 190]
    if side == "ego_left":
        return [40, 230, 90, 200]
    return [255, 150, 40, 200]


def load_depth(depth_dir: Path, frame_idx: int) -> np.ndarray:
    path = depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def fitted_mesh_from_observation(obs: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = Path(str(obs["mesh_obj"]))
    vertices_obj, faces, _ = parse_obj_mesh(mesh_path)
    vertices_camera = hamer_rendered_obj_to_camera_opencv(vertices_obj)
    fitted = translate_points(vertices_camera, obs["smoothed_translation"])
    return fitted, faces


def draw_mesh_on_rgb(
    image_rgb: np.ndarray,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    color_rgb: List[int],
    radius: int = 1,
) -> np.ndarray:
    out = image_rgb.copy()
    pixels, valid = project_points_to_image(
        vertices_camera,
        ROPEDIA_K,
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


def write_tracks_json(
    tracks: Dict[str, List[Optional[Dict[str, Any]]]],
    summary: Dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    payload = {
        "inputs": {
            "alignment_report": str(Path(args.alignment_report).expanduser().resolve()),
            "depth_dir": str(Path(args.depth_dir).expanduser().resolve()),
            "video": str(Path(args.video).expanduser().resolve()),
        },
        "parameters": {
            "start": args.start,
            "num_frames": args.num_frames,
            "max_interp_gap": args.max_interp_gap,
            "smooth_alpha": args.smooth_alpha,
            "jump_threshold_m": args.jump_threshold_m,
            "mesh_error_threshold_m": args.mesh_error_threshold_m,
            "min_valid_depth_ratio": args.min_valid_depth_ratio,
            "drop_unstable_recovered": args.drop_unstable_recovered,
            "unstable_recovered_jump_threshold_m": args.unstable_recovered_jump_threshold_m,
            "unstable_recovered_max_area_ratio": args.unstable_recovered_max_area_ratio,
        },
        "summary": summary,
        "tracks": jsonable(tracks),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_tracks_csv(tracks: Dict[str, List[Optional[Dict[str, Any]]]], output_path: Path, start: int) -> None:
    fieldnames = [
        "frame_idx",
        "side",
        "status",
        "candidate_id",
        "mesh_obj",
        "translation_x",
        "translation_y",
        "translation_z",
        "smoothed_translation_x",
        "smoothed_translation_y",
        "smoothed_translation_z",
        "raw_translation_jump_m",
        "smoothed_translation_jump_m",
        "mesh_to_depth_median_m",
        "depth_to_mesh_median_m",
        "anomaly_flags",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for side, track in tracks.items():
            for offset, obs in enumerate(track):
                frame_idx = start + offset
                if obs is None:
                    writer.writerow({"frame_idx": frame_idx, "side": side, "status": "missing"})
                    continue
                tr = obs["translation"]
                sm = obs["smoothed_translation"]
                writer.writerow(
                    {
                        "frame_idx": frame_idx,
                        "side": side,
                        "status": obs.get("status"),
                        "candidate_id": obs.get("candidate_id"),
                        "mesh_obj": obs.get("mesh_obj"),
                        "translation_x": tr[0],
                        "translation_y": tr[1],
                        "translation_z": tr[2],
                        "smoothed_translation_x": sm[0],
                        "smoothed_translation_y": sm[1],
                        "smoothed_translation_z": sm[2],
                        "raw_translation_jump_m": obs.get("raw_translation_jump_m"),
                        "smoothed_translation_jump_m": obs.get("smoothed_translation_jump_m"),
                        "mesh_to_depth_median_m": obs.get("mesh_to_depth_median_m"),
                        "depth_to_mesh_median_m": obs.get("depth_to_mesh_median_m"),
                        "anomaly_flags": ",".join(obs.get("anomaly_flags", [])),
                    }
                )


def draw_debug_frame(rgb: np.ndarray, frame_idx: int, observations: Dict[str, Optional[Dict[str, Any]]]) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    y = 22
    cv2.putText(out, f"frame {frame_idx}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 24
    for side in SIDES:
        obs = observations.get(side)
        if obs is None:
            label = f"{side}: missing"
            color = (0, 0, 255)
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in obs.get("bbox_xyxy", [0, 0, 0, 0])]
            has_anomaly = bool([f for f in obs.get("anomaly_flags", []) if f != "interpolated"])
            rgba = status_color_rgba(side, obs.get("status", ""), has_anomaly)
            color = (rgba[2], rgba[1], rgba[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = (
                f"{side}: {obs.get('status')} "
                f"jump={obs.get('smoothed_translation_jump_m')} "
                f"flags={','.join(obs.get('anomaly_flags', [])) or 'ok'}"
            )
        cv2.putText(out, label[:95], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 20
    return out


def make_contact_sheet(image_paths: List[Path], output_path: Path, columns: int, thumb_width: int) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = thumb_width / float(image.shape[1])
        thumbs.append(cv2.resize(image, (thumb_width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    rows = int(np.ceil(len(thumbs) / columns))
    thumb_h = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows * thumb_h, columns * thumb_width, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r = i // columns
        c = i % columns
        y = r * thumb_h
        x = c * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    cv2.imwrite(str(output_path), sheet)


def export_debug_contact_sheet(
    tracks: Dict[str, List[Optional[Dict[str, Any]]]],
    video_path: Path,
    output_dir: Path,
    start: int,
    max_debug_frames: int,
) -> None:
    debug_dir = output_dir / "debug_frames"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    selected_offsets = []
    for offset in range(len(next(iter(tracks.values())))):
        observations = {side: tracks[side][offset] for side in SIDES}
        important = any(obs is None for obs in observations.values())
        important = important or any(
            obs is not None and (obs.get("status") == "interpolated" or obs.get("anomaly_flags"))
            for obs in observations.values()
        )
        if important or offset % 30 == 0:
            selected_offsets.append(offset)
    selected_offsets = selected_offsets[:max_debug_frames]

    paths = []
    for offset in selected_offsets:
        frame_idx = start + offset
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        panel = draw_debug_frame(rgb, frame_idx, {side: tracks[side][offset] for side in SIDES})
        out_path = debug_dir / f"frame_{frame_idx:06d}_tracking_debug.jpg"
        cv2.imwrite(str(out_path), panel)
        paths.append(out_path)
    make_contact_sheet(paths, output_dir / "contact_sheet_tracking_debug.jpg", columns=4, thumb_width=360)


def log_rerun(
    tracks: Dict[str, List[Optional[Dict[str, Any]]]],
    video_path: Path,
    depth_dir: Path,
    save_rrd: Path,
    args: argparse.Namespace,
) -> None:
    rr.init("Ropedia Ego Hand Tracking Smoothed", spawn=False)
    rr.save(str(save_rrd))
    rr.log("camera_display", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("camera_opencv", rr.ViewCoordinates.RDF, static=True)
    for offset in range(args.num_frames):
        frame_idx = args.start + offset
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        depth_m = load_depth(depth_dir, frame_idx)
        rr.set_time("frame", sequence=frame_idx)
        rr.log("camera_display/hands", rr.Clear(recursive=True))
        rr.log("camera_display/camera/left", rr.Transform3D(translation=[0, 0, 0]))
        log_pinhole("camera_display/camera/left", ROPEDIA_K, rgb.shape[1], rgb.shape[0])
        rr.log("camera_display/camera/left/rgb", rr.Image(rgb))
        rr.log("camera_display/camera/left/depth_vis", rr.Image(depth_to_vis(depth_m, args.min_depth_m, args.max_depth_m)))
        rr.log("camera_opencv/hands", rr.Clear(recursive=True))
        rr.log("camera_opencv/camera/left", rr.Transform3D(translation=[0, 0, 0]))
        log_pinhole("camera_opencv/camera/left", ROPEDIA_K, rgb.shape[1], rgb.shape[0])
        rr.log("camera_opencv/camera/left/rgb", rr.Image(rgb))
        rr.log("camera_opencv/camera/left/depth_vis", rr.Image(depth_to_vis(depth_m, args.min_depth_m, args.max_depth_m)))
        rgb_mesh_overlay = rgb.copy()
        if args.log_full_scene:
            points, colors = depth_to_pointcloud(
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
                rr.Points3D(opencv_to_rerun_display(points), colors=colors, radii=args.point_radius),
            )
            rr.log(
                "camera_opencv/depth/full_scene_points",
                rr.Points3D(points, colors=colors, radii=args.point_radius),
            )
        for side in SIDES:
            obs = tracks[side][offset]
            if obs is None:
                continue
            vertices_camera, faces = fitted_mesh_from_observation(obs)
            vertices_display = opencv_to_rerun_display(vertices_camera)
            non_interp_anomaly = bool([f for f in obs.get("anomaly_flags", []) if f != "interpolated"])
            color = status_color_rgba(side, obs.get("status", ""), non_interp_anomaly)
            rgb_mesh_overlay = draw_mesh_on_rgb(
                rgb_mesh_overlay,
                vertices_camera,
                faces,
                color_rgb=color[:3],
                radius=max(1, int(round(args.rgb_overlay_point_radius))),
            )
            draw_bbox_and_label(rgb_mesh_overlay, obs, side, color_rgb=color[:3])
            rr.log(
                f"camera_display/hands/{side}/current_best_mesh_smoothed",
                rr.Mesh3D(
                    vertex_positions=vertices_display,
                    triangle_indices=faces,
                    albedo_factor=color,
                ),
            )
            rr.log(
                f"camera_opencv/hands/{side}/current_best_mesh_smoothed",
                rr.Mesh3D(
                    vertex_positions=vertices_camera,
                    triangle_indices=faces,
                    albedo_factor=color,
                ),
            )
            centroid = np.median(vertices_display, axis=0)
            rr.log(
                f"camera_display/hands/{side}/status",
                rr.Points3D(
                    centroid.reshape(1, 3),
                    colors=np.array([status_color_rgba(side, obs.get("status", ""), non_interp_anomaly)[:3]], dtype=np.uint8),
                    radii=args.point_radius * 3.0,
                ),
            )
        rr.log("camera_display/camera/left/rgb_mesh_overlay", rr.Image(rgb_mesh_overlay))
        rr.log("camera_opencv/camera/left/rgb_mesh_overlay", rr.Image(rgb_mesh_overlay))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track, interpolate, smooth, and flag ego hand trajectories.")
    parser.add_argument("--alignment_report", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--video", default="/home/liuyou/Documents/ropedia_data/stereo_left.mp4")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--max_interp_gap", type=int, default=5)
    parser.add_argument("--smooth_alpha", type=float, default=0.35)
    parser.add_argument("--jump_threshold_m", type=float, default=0.08)
    parser.add_argument("--mesh_error_threshold_m", type=float, default=0.05)
    parser.add_argument("--min_valid_depth_ratio", type=float, default=0.5)
    parser.add_argument(
        "--drop_unstable_recovered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop recovered low-area candidates when their 3D anchor jumps too far from neighboring frames.",
    )
    parser.add_argument("--unstable_recovered_jump_threshold_m", type=float, default=0.08)
    parser.add_argument(
        "--unstable_recovered_max_area_ratio",
        type=float,
        default=0.001,
        help="Only temporal-gate recovered candidates below this bbox area ratio. Use <=0 to gate all recovered candidates.",
    )
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=2.5)
    parser.add_argument("--log_full_scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene_point_stride", type=int, default=6)
    parser.add_argument("--max_scene_points", type=int, default=50000)
    parser.add_argument("--point_radius", type=float, default=0.004)
    parser.add_argument("--rgb_overlay_point_radius", type=float, default=1.0)
    parser.add_argument("--max_debug_frames", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    alignment_report = resolve_path(args.alignment_report, cwd)
    depth_dir = resolve_path(args.depth_dir, cwd)
    video_path = resolve_path(args.video, cwd)
    output_dir = resolve_path(args.output_dir, cwd)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(alignment_report.read_text())
    tracks = process_tracks(
        report["frames"],
        start=args.start,
        num_frames=args.num_frames,
        max_interp_gap=args.max_interp_gap,
        smooth_alpha=args.smooth_alpha,
        jump_threshold_m=args.jump_threshold_m,
        mesh_error_threshold_m=args.mesh_error_threshold_m,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        drop_unstable_recovered=args.drop_unstable_recovered,
        unstable_recovered_jump_threshold_m=args.unstable_recovered_jump_threshold_m,
        unstable_recovered_max_area_ratio=(
            None if args.unstable_recovered_max_area_ratio <= 0 else args.unstable_recovered_max_area_ratio
        ),
    )
    summary = track_summary(tracks)
    write_tracks_json(tracks, summary, output_dir / "ego_hand_tracks.json", args)
    write_tracks_csv(tracks, output_dir / "ego_hand_tracks.csv", start=args.start)
    export_debug_contact_sheet(tracks, video_path, output_dir, args.start, args.max_debug_frames)
    log_rerun(
        tracks,
        video_path=video_path,
        depth_dir=depth_dir,
        save_rrd=output_dir / "ropedia_ego_hands_tracked_smoothed.rrd",
        args=args,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
