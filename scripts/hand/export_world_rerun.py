#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.orientation import load_mano_regressors  # noqa: E402
from roboego_hand_vis.egohand.rerun_logging import log_mesh3d  # noqa: E402
from roboego_hand_vis.egohand.temporal_tracking import SIDES  # noqa: E402
from roboego_hand_vis.trajectory import RopediaSlamTrajectory  # noqa: E402
from scripts.export_depth_pointcloud import depth_to_pointcloud, read_rgb_frame  # noqa: E402
from scripts.hand.export_final_rerun import iter_export_offsets, load_depth  # noqa: E402
from scripts.rerun_utils import depth_to_vis, log_pinhole  # noqa: E402
from scripts.track_and_smooth_ego_hands import build_final_smoothed_meshes, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export final estimated ego hand meshes and depth point clouds in a world frame. "
            "Currently supports the validated Ropedia direct_T_cam_world SLAM convention."
        )
    )
    parser.add_argument("--tracks_json", required=True)
    parser.add_argument("--alignment_report", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_rrd", required=True)
    parser.add_argument("--manifest", default="", help="Optional dataset manifest used to find Ropedia annotation.hdf5.")
    parser.add_argument("--annotation_hdf5", default="", help="Optional explicit Ropedia annotation.hdf5 path.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--rrd_start", type=int, default=None)
    parser.add_argument("--rrd_num_frames", type=int, default=0)
    parser.add_argument("--final_smoothing_window", type=int, default=None)
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=2.5)
    parser.add_argument("--scene_point_stride", type=int, default=8)
    parser.add_argument("--max_scene_points", type=int, default=30000)
    parser.add_argument("--accumulated_point_stride", type=int, default=16)
    parser.add_argument("--max_accumulated_points_per_frame", type=int, default=6000)
    parser.add_argument("--point_radius", type=float, default=0.004)
    parser.add_argument("--world_point_radius", type=float, default=0.004)
    parser.add_argument("--mesh_color_alpha", type=int, default=220)
    parser.add_argument("--mano_model_dir", default="")
    parser.add_argument(
        "--log_camera_space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also log RGB/depth/final mesh in camera_opencv for direct comparison.",
    )
    parser.add_argument(
        "--log_accumulated_world_points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log a static accumulated world point cloud at the final frame.",
    )
    return parser.parse_args()


def annotation_from_manifest(manifest_path: Path | None) -> Path | None:
    if manifest_path is None:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = (manifest.get("annotations") or {}).get("annotation_hdf5")
    return Path(value).expanduser().resolve() if value else None


def resolve_annotation(args: argparse.Namespace, cwd: Path) -> Path:
    if args.annotation_hdf5:
        return resolve_path(args.annotation_hdf5, cwd)
    manifest_path = resolve_path(args.manifest, cwd) if args.manifest else None
    annotation = annotation_from_manifest(manifest_path)
    if annotation is None:
        raise ValueError("Provide --annotation_hdf5 or a Ropedia --manifest with annotations.annotation_hdf5")
    return annotation


def default_range(args: argparse.Namespace, tracks_payload: dict) -> tuple[int, int, int]:
    parameters = tracks_payload.get("parameters") or {}
    start = args.start if args.start is not None else int(parameters.get("start", 0))
    num_frames = args.num_frames if args.num_frames is not None else int(parameters.get("num_frames", 0))
    if num_frames <= 0:
        tracks = tracks_payload["tracks"]
        num_frames = max(len(track) for track in tracks.values())
    window = (
        args.final_smoothing_window
        if args.final_smoothing_window is not None
        else int(parameters.get("final_smoothing_window", 5))
    )
    return start, num_frames, window


def load_regressors(args: argparse.Namespace, cwd: Path, orientation_config: dict) -> dict[str, np.ndarray]:
    mano_model_dir = (
        resolve_path(args.mano_model_dir, cwd)
        if args.mano_model_dir
        else Path(orientation_config.get("mano_model_dir") or "").expanduser()
    )
    if orientation_config.get("enabled") and str(mano_model_dir):
        return load_mano_regressors(mano_model_dir)
    return {}


def transform_mesh_to_world(vertices: np.ndarray, pose) -> np.ndarray:
    return pose.camera_to_world(vertices)


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    tracks_json = resolve_path(args.tracks_json, cwd)
    alignment_report = resolve_path(args.alignment_report, cwd)
    depth_dir = resolve_path(args.depth_dir, cwd)
    output_rrd = resolve_path(args.output_rrd, cwd)
    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    annotation_hdf5 = resolve_annotation(args, cwd)

    tracks_payload = json.loads(tracks_json.read_text(encoding="utf-8"))
    alignment_payload = json.loads(alignment_report.read_text(encoding="utf-8"))
    start, num_frames, final_window = default_range(args, tracks_payload)
    args.start = start
    args.num_frames = num_frames
    tracks = tracks_payload["tracks"]
    video_path = Path(tracks_payload["video"]).expanduser().resolve()
    intrinsics = alignment_payload["intrinsics"]
    orientation_config = alignment_payload.get("orientation_aware_compensation") or {"enabled": False}
    regressor_by_side = load_regressors(args, cwd, orientation_config)
    final_tracks = build_final_smoothed_meshes(
        tracks,
        window=final_window,
        regressor_by_side=regressor_by_side,
        orientation_config=orientation_config,
    )
    trajectory = RopediaSlamTrajectory(annotation_hdf5)

    rr.init("RoboEgo Estimated Hands World Vis", spawn=False)
    rr.save(str(output_rrd))
    rr.log("camera_opencv", rr.ViewCoordinates.RDF, static=True)
    rr.log("world_direct_T_cam_world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "notes",
        rr.TextDocument(
            "Estimated hand/depth world export. Uses Ropedia direct_T_cam_world convention: "
            "SLAM pose is T_camera_world, so P_world = inverse(T_camera_world) @ P_camera."
        ),
    )

    accumulated_points: list[np.ndarray] = []
    accumulated_colors: list[np.ndarray] = []
    camera_centers: list[np.ndarray] = []
    frames_written = 0
    hands_written = {side: 0 for side in SIDES}

    for offset, frame_idx in iter_export_offsets(args):
        pose = trajectory.pose(frame_idx)
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        depth_m = load_depth(depth_dir, frame_idx)
        rr.set_time("frame", sequence=frame_idx)

        if args.log_camera_space:
            rr.log("camera_opencv/camera/left", rr.Transform3D(translation=[0, 0, 0]))
            log_pinhole("camera_opencv/camera/left", intrinsics, rgb.shape[1], rgb.shape[0])
            rr.log("camera_opencv/camera/left/rgb", rr.Image(rgb))
            rr.log("camera_opencv/camera/left/depth_vis", rr.Image(depth_to_vis(depth_m, args.min_depth_m, args.max_depth_m)))

        points_camera, colors = depth_to_pointcloud(
            depth_m=depth_m,
            rgb=rgb,
            intrinsics=intrinsics,
            stride=max(1, args.scene_point_stride),
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            max_points=args.max_scene_points,
        )
        points_world = pose.camera_to_world(points_camera)
        if args.log_camera_space:
            rr.log("camera_opencv/depth/full_scene_points", rr.Points3D(points_camera, colors=colors, radii=args.point_radius))
        rr.log("world_direct_T_cam_world/depth/current_points", rr.Points3D(points_world, colors=colors, radii=args.world_point_radius))

        if args.log_accumulated_world_points:
            acc_camera, acc_colors = depth_to_pointcloud(
                depth_m=depth_m,
                rgb=rgb,
                intrinsics=intrinsics,
                stride=max(1, args.accumulated_point_stride),
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                max_points=args.max_accumulated_points_per_frame,
            )
            accumulated_points.append(pose.camera_to_world(acc_camera))
            accumulated_colors.append(acc_colors)

        camera_centers.append(pose.camera_center_world.astype(np.float32))
        rr.log(
            "world_direct_T_cam_world/camera/current_center",
            rr.Points3D([pose.camera_center_world.astype(np.float32)], colors=[255, 255, 255], radii=0.03),
        )

        for side in SIDES:
            item = final_tracks.get(side, [None] * num_frames)[offset]
            if item is None:
                continue
            color = [60, 255, 120, args.mesh_color_alpha] if side == "ego_left" else [80, 190, 255, args.mesh_color_alpha]
            vertices_camera = np.asarray(item["vertices"], dtype=np.float32)
            faces = np.asarray(item["faces"], dtype=np.uint32)
            vertices_world = transform_mesh_to_world(vertices_camera, pose)
            if args.log_camera_space:
                log_mesh3d(f"camera_opencv/hands/{side}/final", vertices_camera, faces, color)
            log_mesh3d(f"world_direct_T_cam_world/hands/{side}/final", vertices_world, faces, color)
            hands_written[side] += 1

        frames_written += 1

    if camera_centers:
        centers = np.asarray(camera_centers, dtype=np.float32)
        rr.set_time("frame", sequence=start + frames_written - 1)
        rr.log("world_direct_T_cam_world/camera/trajectory", rr.LineStrips3D([centers], colors=[255, 255, 255], radii=0.01))
        rr.log("world_direct_T_cam_world/camera/centers", rr.Points3D(centers, colors=[255, 255, 255], radii=0.018))
    if args.log_accumulated_world_points and accumulated_points:
        rr.log(
            "world_direct_T_cam_world/depth/accumulated_points",
            rr.Points3D(
                np.concatenate(accumulated_points, axis=0),
                colors=np.concatenate(accumulated_colors, axis=0),
                radii=max(args.world_point_radius * 0.7, 0.001),
            ),
        )

    report_path = output_rrd.with_suffix(".json")
    report = {
        "output_rrd": str(output_rrd),
        "tracks_json": str(tracks_json),
        "alignment_report": str(alignment_report),
        "depth_dir": str(depth_dir),
        "annotation_hdf5": str(annotation_hdf5),
        "trajectory_convention": "ropedia_direct_T_cam_world",
        "camera_to_world": "P_world = inverse(T_camera_world) @ P_camera",
        "start": start,
        "num_frames": num_frames,
        "frames_written": frames_written,
        "final_smoothing_window": final_window,
        "hands_written": hands_written,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Rerun: {output_rrd}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
