#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.objects.annotation_io import write_json_payload, write_object_tracks_csv, write_object_tracks_json
from roboego_hand_vis.objects.contact import contact_payload_from_joints_npz_and_object_points
from roboego_hand_vis.objects.color_yellow import YellowSegmentationConfig, detect_yellow_object, fill_missing_observations, save_mask_and_crop
from roboego_hand_vis.objects.depth_geometry import load_depth_frame, mask_depth_to_points
from roboego_hand_vis.objects.rerun_export import draw_object_bbox_overlay, log_object_overlay
from roboego_hand_vis.objects.semantic_qwen import call_qwen_semantics, select_key_frames


def iter_rgb_frames(video_path: Path, num_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    try:
        for frame_idx in range(num_frames):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
            yield frame_idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _sanitized_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if "DASHSCOPE_API_KEY" in message:
        return "DASHSCOPE_API_KEY is not set"
    return f"Qwen request failed ({type(exc).__name__})."


def log_rerun_overlays(
    *,
    video_path: Path,
    observations: list,
    object_id: str,
    output_dir: Path,
    jpeg_quality: int,
) -> None:
    rr.init("RoboEgo Object Annotation", spawn=False)
    rr.save(str(output_dir / "object_annotation.rrd"))
    for (frame_idx, rgb), obs in zip(iter_rgb_frames(video_path, len(observations)), observations):
        rr.set_time_sequence("frame", frame_idx)
        overlay = draw_object_bbox_overlay(rgb, obs, label=object_id)
        log_object_overlay(
            "camera_opencv/camera/left/rgb_object_bbox_overlay",
            overlay,
            jpeg_quality=jpeg_quality,
        )


def run_annotation(
    *,
    video_path: Path,
    depth_dir: Path,
    output_dir: Path,
    intrinsics: dict[str, float],
    hand_tracks_path: Path | None,
    hand_joints_npz: Path | None,
    backend: str,
    object_id: str,
    num_frames: int,
    no_qwen: bool,
    save_rrd: bool,
    qwen_model: str = "qwen-vl-plus",
    contact_threshold_m: float = 0.025,
    jpeg_quality: int = 85,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = []
    object_points_by_frame: dict[int, np.ndarray] = {}
    warnings: list[dict[str, Any]] = []
    config = YellowSegmentationConfig()
    if backend != "color_yellow":
        raise ValueError(f"Unsupported backend in this CLI pass: {backend}")
    for frame_idx, rgb in iter_rgb_frames(video_path, num_frames):
        obs, mask = detect_yellow_object(rgb, frame_idx=frame_idx, object_id=object_id, config=config)
        obs = save_mask_and_crop(image_rgb=rgb, mask=mask, obs=obs, output_dir=output_dir, crop_padding_px=12)
        if obs.bbox_xywh is not None:
            try:
                depth = load_depth_frame(depth_dir, frame_idx)
            except FileNotFoundError:
                warnings.append({"frame_idx": frame_idx, "status": "depth_file_missing"})
            else:
                points, stats = mask_depth_to_points(
                    depth,
                    mask,
                    intrinsics=intrinsics,
                    min_depth_m=0.15,
                    max_depth_m=2.5,
                    stride=4,
                )
                object_points_by_frame[frame_idx] = points
                obs.depth_median_m = stats.depth_median_m
                obs.depth_valid_ratio = stats.depth_valid_ratio
        observations.append(obs)
    observations = fill_missing_observations(observations, max_gap=config.gap_fill_frames)
    if save_rrd:
        log_rerun_overlays(
            video_path=video_path,
            observations=observations,
            object_id=object_id,
            output_dir=output_dir,
            jpeg_quality=jpeg_quality,
        )
    write_object_tracks_json(
        observations={object_id: observations},
        output_path=output_dir / "object_tracks.json",
        output_dir=output_dir,
        backend=backend,
        video_path=video_path,
        depth_dir=depth_dir,
        warnings=warnings,
    )
    write_object_tracks_csv({object_id: observations}, output_dir / "object_tracks.csv")

    semantic = {"schema_version": "1.0", "provider": "none", "object_id": object_id, "status": "skipped"}
    if not no_qwen:
        key_frames = select_key_frames(observations, max_frames=4)
        crop_paths = [obs.crop_path for obs in observations if obs.frame_idx in key_frames and obs.crop_path is not None]
        if not crop_paths:
            semantic = {
                "schema_version": "1.0",
                "provider": "qwen",
                "model": qwen_model,
                "object_id": object_id,
                "evidence_frames": key_frames,
                "status": "no_crops",
            }
        else:
            try:
                semantic = call_qwen_semantics(crop_paths=crop_paths, model=qwen_model)
            except Exception as exc:
                semantic = {
                    "schema_version": "1.0",
                    "provider": "qwen",
                    "model": qwen_model,
                    "object_id": object_id,
                    "evidence_frames": key_frames,
                    "status": "api_failed",
                    "error_type": type(exc).__name__,
                    "error": _sanitized_error(exc),
                }
            else:
                semantic.update({"schema_version": "1.0", "object_id": object_id, "evidence_frames": key_frames})
    write_json_payload(semantic, output_dir / "semantic_annotations.json")

    if hand_joints_npz is not None and hand_joints_npz.exists():
        contact_payload = contact_payload_from_joints_npz_and_object_points(
            object_id=object_id,
            hand_joints_npz=hand_joints_npz,
            object_points_by_frame=object_points_by_frame,
            threshold_m=contact_threshold_m,
            frame_count=len(observations),
        )
    else:
        contact_payload = {
            "schema_version": "1.0",
            "object_id": object_id,
            "distance_threshold_m": contact_threshold_m,
            "hand_tracks": str(hand_tracks_path) if hand_tracks_path else None,
            "hand_joints_npz": str(hand_joints_npz) if hand_joints_npz else None,
            "status": "hand_joints_npz_missing",
            "frames": [
                {
                    "frame_idx": obs.frame_idx,
                    "left_contact": False,
                    "right_contact": False,
                    "left_min_distance_m": None,
                    "right_min_distance_m": None,
                    "contact_points": [],
                    "status": "hand_joints_npz_missing",
                }
                for obs in observations
            ],
        }
    write_json_payload(contact_payload, output_dir / "contact_annotations.json")

    return {"object_id": object_id, "frame_count": num_frames, "output_dir": str(output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment and annotate manipulated objects in ego video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--intrinsics_json", default="tmp/baai_rectified_intrinsics.json")
    parser.add_argument("--hand_tracks", default="")
    parser.add_argument("--hand_joints_npz", default="")
    parser.add_argument("--backend", default="color_yellow", choices=["color_yellow"])
    parser.add_argument("--object_id", default="yellow_object_0")
    parser.add_argument("--num_frames", type=int, default=900)
    parser.add_argument("--no_qwen", action="store_true")
    parser.add_argument("--qwen_model", default="qwen-vl-plus")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--no_save_rrd", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intrinsics_data = json.loads(Path(args.intrinsics_json).read_text())
    intrinsics = intrinsics_data.get("left", intrinsics_data)
    result = run_annotation(
        video_path=Path(args.video),
        depth_dir=Path(args.depth_dir),
        output_dir=Path(args.output_dir),
        intrinsics=intrinsics,
        hand_tracks_path=Path(args.hand_tracks) if args.hand_tracks else None,
        hand_joints_npz=Path(args.hand_joints_npz) if args.hand_joints_npz else None,
        backend=args.backend,
        object_id=args.object_id,
        num_frames=args.num_frames,
        no_qwen=args.no_qwen,
        save_rrd=not args.no_save_rrd,
        qwen_model=args.qwen_model,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
