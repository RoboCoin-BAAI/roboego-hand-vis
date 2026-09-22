#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.hand_depth_alignment.orientation import load_mano_regressors  # noqa: E402
from roboego_hand_vis.hand_rerun_visualization.rerun_logging import log_mesh3d, log_points2d, rgb_overlay_entity, side_mesh_entity  # noqa: E402
from roboego_hand_vis.hand_rerun_visualization.temporal_tracking import SIDES  # noqa: E402
from roboego_hand_vis.common.depth_io import load_depth_frame  # noqa: E402
from scripts.export_depth_pointcloud import ROPEDIA_K, depth_to_pointcloud, read_rgb_frame  # noqa: E402
from scripts.rerun_utils import depth_to_vis, log_pinhole  # noqa: E402
from scripts.track_and_smooth_ego_hands import (  # noqa: E402
    build_final_smoothed_meshes,
    project_points_to_image,
    resolve_path,
)


def iter_export_offsets(args: argparse.Namespace) -> Iterator[tuple[int, int]]:
    rrd_start = args.start if args.rrd_start is None else args.rrd_start
    if rrd_start < args.start:
        raise ValueError(f"--rrd_start {rrd_start} is before --start {args.start}")
    first_offset = rrd_start - args.start
    if first_offset >= args.num_frames:
        raise ValueError(
            f"--rrd_start {rrd_start} is outside processed range "
            f"[{args.start}, {args.start + args.num_frames - 1}]"
        )
    count = args.num_frames - first_offset if args.rrd_num_frames <= 0 else args.rrd_num_frames
    for offset in range(first_offset, min(args.num_frames, first_offset + count)):
        yield offset, args.start + offset


def load_depth(depth_dir: Path, frame_idx: int) -> np.ndarray:
    return load_depth_frame(depth_dir, frame_idx).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a clean final Rerun file with RGB, point cloud, and final meshes.")
    parser.add_argument("--tracks_json", required=True)
    parser.add_argument("--alignment_report", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_rrd", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=600)
    parser.add_argument("--rrd_start", type=int, default=None)
    parser.add_argument("--rrd_num_frames", type=int, default=0)
    parser.add_argument("--final_smoothing_window", type=int, default=5)
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=2.5)
    parser.add_argument("--scene_point_stride", type=int, default=8)
    parser.add_argument("--max_scene_points", type=int, default=30000)
    parser.add_argument("--point_radius", type=float, default=0.004)
    parser.add_argument("--mesh_point_radius", type=float, default=1.0)
    parser.add_argument("--mano_model_dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    tracks_json = resolve_path(args.tracks_json, cwd)
    alignment_report = resolve_path(args.alignment_report, cwd)
    depth_dir = resolve_path(args.depth_dir, cwd)
    output_rrd = resolve_path(args.output_rrd, cwd)
    output_rrd.parent.mkdir(parents=True, exist_ok=True)

    tracks_payload = json.loads(tracks_json.read_text(encoding="utf-8"))
    alignment_payload = json.loads(alignment_report.read_text(encoding="utf-8"))
    tracks = tracks_payload["tracks"]
    video_path = Path(tracks_payload["video"]).expanduser().resolve()
    intrinsics = alignment_payload.get("intrinsics") or dict(ROPEDIA_K)
    orientation_config = alignment_payload.get("orientation_aware_compensation") or {"enabled": False}
    mano_model_dir = (
        resolve_path(args.mano_model_dir, cwd)
        if args.mano_model_dir
        else Path(orientation_config.get("mano_model_dir") or "").expanduser()
    )
    regressor_by_side = {}
    if orientation_config.get("enabled") and str(mano_model_dir):
        regressor_by_side = load_mano_regressors(mano_model_dir)

    final_tracks = build_final_smoothed_meshes(
        tracks,
        window=args.final_smoothing_window,
        regressor_by_side=regressor_by_side,
        orientation_config=orientation_config,
    )

    rr.init("RoboEgo Final Hand Vis", spawn=False)
    rr.save(str(output_rrd))
    rr.log("camera_opencv", rr.ViewCoordinates.RDF, static=True)

    for offset, frame_idx in iter_export_offsets(args):
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        depth_m = load_depth(depth_dir, frame_idx)
        rr.set_time("frame", sequence=frame_idx)
        rr.log("camera_opencv/camera/left", rr.Transform3D(translation=[0, 0, 0]))
        log_pinhole("camera_opencv/camera/left", intrinsics, rgb.shape[1], rgb.shape[0])
        rr.log("camera_opencv/camera/left/rgb", rr.Image(rgb))
        rr.log("camera_opencv/camera/left/depth_vis", rr.Image(depth_to_vis(depth_m, args.min_depth_m, args.max_depth_m)))

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
            item = final_tracks.get(side, [None] * args.num_frames)[offset]
            if item is None:
                continue
            vertices = item["vertices"]
            faces = item["faces"]
            color = [60, 255, 120, 220] if side == "ego_left" else [80, 190, 255, 220]
            log_mesh3d(side_mesh_entity(side, "final"), vertices, faces, color)
            pixels, _ = project_points_to_image(vertices, intrinsics, rgb.shape[1], rgb.shape[0])
            if len(pixels) > 0:
                log_points2d(
                    rgb_overlay_entity("final", side, "mesh_projected_points"),
                    pixels,
                    color,
                    radius=max(1.0, float(args.mesh_point_radius)),
                )

    print(f"Rerun: {output_rrd}")


if __name__ == "__main__":
    main()
