#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pyarrow.parquet as pq
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.common.depth_io import load_depth_frame  # noqa: E402
from roboego_hand_vis.hand_rerun_visualization.joints21 import HAND21_BONES, joints_to_lines  # noqa: E402
from linshi.scripts.export_depth_pointcloud import depth_to_pointcloud, read_rgb_frame  # noqa: E402
from linshi.scripts.rerun_utils import depth_to_vis, log_pinhole  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export depth point cloud plus final 21-joint hand skeletons to Rerun.")
    parser.add_argument("--joints_npz", default="")
    parser.add_argument("--hand_tracking_parquet", default="")
    parser.add_argument("--video", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_rrd", required=True)
    parser.add_argument("--intrinsics_json", default="", help="fast_foundation_stereo_video_meta.json; defaults under --depth_dir.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=0, help="Processed frame count. <=0 derives from joints npz.")
    parser.add_argument("--rrd_start", type=int, default=None)
    parser.add_argument("--rrd_num_frames", type=int, default=1)
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=2.5)
    parser.add_argument("--scene_point_stride", type=int, default=4)
    parser.add_argument("--max_scene_points", type=int, default=80000)
    parser.add_argument("--point_radius", type=float, default=0.004)
    parser.add_argument("--joint_radius_m", type=float, default=0.005)
    parser.add_argument("--bone_radius_m", type=float, default=0.0025)
    parser.add_argument("--rgb_joint_radius_px", type=float, default=2.0)
    parser.add_argument("--rgb_bone_radius_px", type=float, default=1.0)
    return parser.parse_args()


def load_intrinsics(depth_dir: Path, intrinsics_json: str | Path = "") -> dict[str, float]:
    path = Path(intrinsics_json).expanduser() if intrinsics_json else depth_dir / "fast_foundation_stereo_video_meta.json"
    if not path.is_absolute():
        path = Path.cwd() / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["intrinsics"]
    return {
        "fx": float(values["fx"]),
        "fy": float(values["fy"]),
        "cx": float(values["cx"]),
        "cy": float(values["cy"]),
    }


def iter_export_offsets(start: int, num_frames: int, rrd_start: int | None, rrd_num_frames: int) -> Iterator[tuple[int, int]]:
    first_frame = start if rrd_start is None else int(rrd_start)
    if first_frame < start:
        raise ValueError(f"--rrd_start {first_frame} is before --start {start}")
    first_offset = first_frame - start
    if first_offset >= num_frames:
        raise ValueError(f"--rrd_start {first_frame} is outside processed range [{start}, {start + num_frames - 1}]")
    count = num_frames - first_offset if rrd_num_frames <= 0 else int(rrd_num_frames)
    for offset in range(first_offset, min(num_frames, first_offset + count)):
        yield offset, start + offset


def side_color(side: str) -> list[int]:
    return [60, 255, 120, 255] if side == "left" else [80, 190, 255, 255]


def project_points(points: np.ndarray, intrinsics: dict[str, float], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(arr).all(axis=1) & (arr[:, 2] > 1e-6)
    pixels = np.full((len(arr), 2), np.nan, dtype=np.float32)
    pixels[valid, 0] = intrinsics["fx"] * arr[valid, 0] / arr[valid, 2] + intrinsics["cx"]
    pixels[valid, 1] = intrinsics["fy"] * arr[valid, 1] / arr[valid, 2] + intrinsics["cy"]
    valid &= (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels, valid


def draw_skeleton_overlay(
    image_rgb: np.ndarray,
    joints: np.ndarray,
    intrinsics: dict[str, float],
    color: list[int],
    joint_radius_px: float,
    bone_radius_px: float,
) -> np.ndarray:
    out = image_rgb.copy()
    pixels, valid = project_points(joints, intrinsics, out.shape[1], out.shape[0])
    color_bgr = (int(color[2]), int(color[1]), int(color[0]))
    for start, end in HAND21_BONES:
        if bool(valid[start]) and bool(valid[end]):
            p0 = tuple(np.round(pixels[start]).astype(np.int32).tolist())
            p1 = tuple(np.round(pixels[end]).astype(np.int32).tolist())
            cv2.line(out, p0, p1, color_bgr, thickness=max(1, int(round(bone_radius_px))), lineType=cv2.LINE_AA)
    for idx, point in enumerate(np.round(pixels).astype(np.int32)):
        if bool(valid[idx]):
            cv2.circle(
                out,
                (int(point[0]), int(point[1])),
                max(1, int(round(joint_radius_px))),
                color_bgr,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
    return out


def npz_side(data: np.lib.npyio.NpzFile, side: str) -> tuple[np.ndarray, np.ndarray]:
    joints = data[f"{side}_joints21_camera_opencv"].astype(np.float32)
    valid = data[f"{side}_valid"].astype(bool)
    return joints, valid


def load_hand_tracking_parquet(path: Path) -> dict[str, dict[str, np.ndarray]]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    hands: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        joints = np.full((len(rows), 21, 3), np.nan, dtype=np.float32)
        valid = np.zeros((len(rows),), dtype=bool)
        for idx, row in enumerate(rows):
            arr = np.asarray(row.get(f"{side}_joints_3d"), dtype=np.float32)
            present = bool(row.get(f"{side}_present"))
            if present and arr.shape == (21, 3) and np.isfinite(arr).all():
                joints[idx] = arr
                valid[idx] = True
        hands[side] = {"joints": joints, "valid": valid}
    return hands


def load_hand_tracks(joints_npz: str, hand_tracking_parquet: str) -> dict[str, dict[str, np.ndarray]]:
    if bool(joints_npz) == bool(hand_tracking_parquet):
        raise ValueError("Provide exactly one of --joints_npz or --hand_tracking_parquet")
    if hand_tracking_parquet:
        return load_hand_tracking_parquet(Path(hand_tracking_parquet).expanduser().resolve())
    data = np.load(Path(joints_npz).expanduser().resolve(), allow_pickle=True)
    left_joints, left_valid = npz_side(data, "left")
    right_joints, right_valid = npz_side(data, "right")
    return {
        "left": {"joints": left_joints, "valid": left_valid},
        "right": {"joints": right_joints, "valid": right_valid},
    }


def log_skeleton(entity_root: str, joints: np.ndarray, color: list[int], joint_radius_m: float, bone_radius_m: float) -> None:
    rr.log(f"{entity_root}/joints21", rr.Points3D(joints, colors=[color] * len(joints), radii=joint_radius_m))
    rr.log(f"{entity_root}/skeleton21", rr.LineStrips3D(joints_to_lines(joints), colors=color, radii=bone_radius_m))


def set_frame_time(frame_idx: int) -> None:
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=frame_idx)
    else:
        rr.set_time_sequence("frame", frame_idx)


def main() -> None:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    depth_dir = Path(args.depth_dir).expanduser().resolve()
    output_rrd = Path(args.output_rrd).expanduser().resolve()
    output_rrd.parent.mkdir(parents=True, exist_ok=True)

    intrinsics = load_intrinsics(depth_dir, args.intrinsics_json)
    hands = load_hand_tracks(args.joints_npz, args.hand_tracking_parquet)
    left_joints, left_valid = hands["left"]["joints"], hands["left"]["valid"]
    right_joints, right_valid = hands["right"]["joints"], hands["right"]["valid"]
    num_frames = int(args.num_frames) if args.num_frames > 0 else len(left_valid)

    rr.init("RoboEgo Depth And Joints21", spawn=False)
    rr.save(str(output_rrd))
    rr.log("camera_opencv", rr.ViewCoordinates.RDF, static=True)

    for offset, frame_idx in iter_export_offsets(args.start, num_frames, args.rrd_start, args.rrd_num_frames):
        rgb = read_rgb_frame(video, frame_idx, resize_scale=1.0)
        depth_m = load_depth_frame(depth_dir, frame_idx).astype(np.float32)
        set_frame_time(frame_idx)
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

        overlay = rgb.copy()
        for side, joints_by_frame, valid_by_frame in (
            ("left", left_joints, left_valid),
            ("right", right_joints, right_valid),
        ):
            if offset >= len(valid_by_frame) or not bool(valid_by_frame[offset]):
                continue
            joints = joints_by_frame[offset]
            if joints.shape != (21, 3) or not np.isfinite(joints).all():
                continue
            color = side_color(side)
            log_skeleton(f"camera_opencv/hands/{side}/final", joints, color, args.joint_radius_m, args.bone_radius_m)
            overlay = draw_skeleton_overlay(
                overlay,
                joints,
                intrinsics,
                color,
                joint_radius_px=args.rgb_joint_radius_px,
                bone_radius_px=args.rgb_bone_radius_px,
            )
        rr.log("camera_opencv/camera/left/rgb_skeleton_overlay/final", rr.Image(overlay))

    print(f"Rerun: {output_rrd}")


if __name__ == "__main__":
    main()
