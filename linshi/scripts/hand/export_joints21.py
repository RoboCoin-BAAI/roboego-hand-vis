#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import cv2
import numpy as np
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.hand_rerun_visualization.joints21 import HAND21_BONES, HAND21_NAMES, joints_to_lines  # noqa: E402
from roboego_hand_vis.hand_depth_alignment.orientation import load_mano_regressors, vertices_to_hamer21_joints  # noqa: E402
from roboego_hand_vis.hand_rerun_visualization.temporal_tracking import SIDES  # noqa: E402
from linshi.scripts.export_depth_pointcloud import ROPEDIA_K, read_rgb_frame  # noqa: E402
from linshi.scripts.rerun_utils import log_pinhole  # noqa: E402
from linshi.scripts.track_and_smooth_ego_hands import build_final_smoothed_meshes, project_points_to_image, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final MANO/HaMeR 21 hand joints and optional skeleton Rerun.")
    parser.add_argument("--tracks_json", required=True)
    parser.add_argument("--alignment_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=900)
    parser.add_argument("--final_smoothing_window", type=int, default=5)
    parser.add_argument("--mano_model_dir", default="")
    parser.add_argument("--save_rrd", default="")
    parser.add_argument("--no_save_rrd", action="store_true")
    parser.add_argument("--rrd_start", type=int, default=None)
    parser.add_argument("--rrd_num_frames", type=int, default=0)
    parser.add_argument("--joint_radius_m", type=float, default=0.01)
    parser.add_argument("--bone_radius_m", type=float, default=0.006)
    parser.add_argument("--rgb_joint_radius_px", type=float, default=3.0)
    parser.add_argument("--rgb_bone_radius_px", type=float, default=2.0)
    return parser.parse_args()


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


def side_color(side: str) -> list[int]:
    return [60, 255, 120, 255] if side == "ego_left" else [80, 190, 255, 255]


def draw_skeleton_overlay(
    image_rgb: np.ndarray,
    joints: np.ndarray,
    intrinsics: Dict[str, float],
    color: list[int],
    joint_radius: float,
    bone_radius: float,
) -> np.ndarray:
    out = image_rgb.copy()
    pixels, valid = project_points_to_image(joints, intrinsics, image_rgb.shape[1], image_rgb.shape[0])
    color_bgr = (int(color[2]), int(color[1]), int(color[0]))
    for start, end in HAND21_BONES:
        if bool(valid[start]) and bool(valid[end]):
            p0 = tuple(np.round(pixels[start]).astype(np.int32).tolist())
            p1 = tuple(np.round(pixels[end]).astype(np.int32).tolist())
            cv2.line(out, p0, p1, color_bgr, thickness=max(1, int(round(bone_radius))), lineType=cv2.LINE_AA)
    for idx, (u, v) in enumerate(np.round(pixels).astype(np.int32)):
        if bool(valid[idx]):
            cv2.circle(out, (int(u), int(v)), max(1, int(round(joint_radius))), color_bgr, thickness=-1, lineType=cv2.LINE_AA)
    return out


def log_skeleton3d(entity_root: str, joints: np.ndarray, color: list[int], joint_radius: float, bone_radius: float) -> None:
    rr.log(f"{entity_root}/joints21", rr.Points3D(joints, colors=[color] * len(joints), radii=joint_radius))
    rr.log(f"{entity_root}/skeleton21", rr.LineStrips3D(joints_to_lines(joints), colors=color, radii=bone_radius))


def export_json(path: Path, joints_by_side: Dict[str, np.ndarray], valid_by_side: Dict[str, np.ndarray], start: int) -> None:
    frames: list[dict[str, Any]] = []
    num_frames = len(next(iter(valid_by_side.values())))
    for offset in range(num_frames):
        frame: dict[str, Any] = {"frame_idx": int(start + offset), "hands": {}}
        for side in SIDES:
            frame["hands"][side] = {
                "valid": bool(valid_by_side[side][offset]),
                "joints21_camera_opencv": (
                    joints_by_side[side][offset].astype(float).tolist() if bool(valid_by_side[side][offset]) else None
                ),
            }
        frames.append(frame)
    path.write_text(
        json.dumps(
            {
                "coordinate_frame": "camera_opencv",
                "unit": "meter",
                "joint_names": HAND21_NAMES,
                "bones": HAND21_BONES,
                "frames": frames,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    tracks_json = resolve_path(args.tracks_json, cwd)
    alignment_report = resolve_path(args.alignment_report, cwd)
    output_dir = resolve_path(args.output_dir, cwd)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks_payload = json.loads(tracks_json.read_text(encoding="utf-8"))
    alignment_payload = json.loads(alignment_report.read_text(encoding="utf-8"))
    tracks = tracks_payload["tracks"]
    video_path = Path(tracks_payload["video"]).expanduser().resolve()
    intrinsics = alignment_payload.get("intrinsics") or dict(ROPEDIA_K)
    orientation_config = alignment_payload.get("orientation_aware_compensation") or {"enabled": False}
    mano_model_dir = (
        resolve_path(args.mano_model_dir, cwd)
        if args.mano_model_dir
        else Path(orientation_config.get("mano_model_dir") or "../hamer/_DATA/data/mano").expanduser()
    )
    regressor_by_side = load_mano_regressors(mano_model_dir)

    final_tracks = build_final_smoothed_meshes(
        tracks,
        window=args.final_smoothing_window,
        regressor_by_side=regressor_by_side,
        orientation_config=orientation_config,
    )

    joints_by_side = {side: np.full((args.num_frames, 21, 3), np.nan, dtype=np.float32) for side in SIDES}
    valid_by_side = {side: np.zeros((args.num_frames,), dtype=bool) for side in SIDES}
    status_by_side = {side: [] for side in SIDES}
    for side in SIDES:
        for offset in range(args.num_frames):
            item = final_tracks.get(side, [None] * args.num_frames)[offset]
            obs = tracks[side][offset] if offset < len(tracks[side]) else None
            status_by_side[side].append(None if obs is None else obs.get("status"))
            if item is None:
                continue
            joints_by_side[side][offset] = vertices_to_hamer21_joints(item["vertices"], regressor_by_side[side])
            valid_by_side[side][offset] = True

    npz_path = output_dir / "joints_21_camera_opencv.npz"
    np.savez_compressed(
        npz_path,
        left_joints_21_camera_opencv=joints_by_side["ego_left"],
        right_joints_21_camera_opencv=joints_by_side["ego_right"],
        left_valid=valid_by_side["ego_left"],
        right_valid=valid_by_side["ego_right"],
        joint_names=np.asarray(HAND21_NAMES),
        bones=np.asarray(HAND21_BONES, dtype=np.int32),
    )
    export_json(output_dir / "joints_21_camera_opencv.json", joints_by_side, valid_by_side, args.start)
    summary = {
        side: {
            "valid": int(valid_by_side[side].sum()),
            "total": int(args.num_frames),
            "valid_ratio": float(valid_by_side[side].mean()),
            "status_counts": {
                str(status): int(status_by_side[side].count(status))
                for status in sorted(set(status_by_side[side]), key=lambda x: "" if x is None else str(x))
            },
        }
        for side in SIDES
    }
    summary["outputs"] = {
        "npz": str(npz_path),
        "json": str(output_dir / "joints_21_camera_opencv.json"),
    }

    if not args.no_save_rrd:
        save_rrd = resolve_path(args.save_rrd, cwd) if args.save_rrd else output_dir / "hand_joints21_skeleton.rrd"
        rr.init("RoboEgo Hand Joints21 Skeleton", spawn=False)
        rr.save(str(save_rrd))
        rr.log("camera_opencv", rr.ViewCoordinates.RDF, static=True)
        for offset, frame_idx in iter_export_offsets(args):
            rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
            rr.set_time("frame", sequence=frame_idx)
            rr.log("camera_opencv/camera/left", rr.Transform3D(translation=[0, 0, 0]))
            log_pinhole("camera_opencv/camera/left", intrinsics, rgb.shape[1], rgb.shape[0])
            overlay = rgb.copy()
            rr.log("camera_opencv/camera/left/rgb", rr.Image(rgb))
            for side in SIDES:
                if not bool(valid_by_side[side][offset]):
                    continue
                color = side_color(side)
                joints = joints_by_side[side][offset]
                log_skeleton3d(f"camera_opencv/hands/{side}/final", joints, color, args.joint_radius_m, args.bone_radius_m)
                overlay = draw_skeleton_overlay(
                    overlay,
                    joints,
                    intrinsics,
                    color,
                    joint_radius=args.rgb_joint_radius_px,
                    bone_radius=args.rgb_bone_radius_px,
                )
            rr.log("camera_opencv/camera/left/rgb_skeleton_overlay/final", rr.Image(overlay))
        summary["outputs"]["rrd"] = str(save_rrd)

    (output_dir / "joints_21_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
