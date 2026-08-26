#!/usr/bin/env python3
"""Export an OpenVINS trajectory NPZ as a standalone Rerun recording."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from roboego_hand_vis.vio.session import load_camera_timestamps


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as trajectory:
        required = {"timestamp_camera", "frame_index", "T_world_camera"}
        missing = required.difference(trajectory.files)
        if missing:
            raise ValueError(
                f"Trajectory is missing arrays: {', '.join(sorted(missing))}"
            )
        timestamps = np.asarray(trajectory["timestamp_camera"], dtype=np.float64)
        frame_indices = np.asarray(trajectory["frame_index"], dtype=np.int64)
        transforms = np.asarray(trajectory["T_world_camera"], dtype=np.float64)
    if timestamps.ndim != 1 or frame_indices.shape != timestamps.shape:
        raise ValueError(
            "Trajectory timestamps and frame indices must be one-dimensional and aligned"
        )
    if transforms.shape != (len(timestamps), 4, 4):
        raise ValueError(
            f"Expected T_world_camera shape ({len(timestamps)}, 4, 4), got {transforms.shape}"
        )
    if not np.isfinite(transforms).all():
        raise ValueError("Trajectory contains non-finite transforms")
    return timestamps, frame_indices, transforms


def log_sampled_camera(
    video_path: Path,
    timestamps_path: Path,
    *,
    sample_fps: float,
    max_width: int,
    jpeg_quality: int,
) -> dict[str, object]:
    timestamps = load_camera_timestamps(timestamps_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_frame_count != len(timestamps):
        capture.release()
        raise ValueError(
            f"Camera video has {source_frame_count} frames but {len(timestamps)} timestamps"
        )
    sample_period = source_fps / sample_fps
    next_sample_frame = 0.0
    source_frame = 0
    samples_written = 0
    try:
        while source_frame < source_frame_count:
            ok = capture.grab()
            if not ok:
                break
            if source_frame + 1e-6 >= next_sample_frame:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not decode camera frame {source_frame}")
                if frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (max_width, round(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not ok:
                    raise RuntimeError(
                        f"Could not JPEG-encode camera frame {source_frame}"
                    )
                rr.set_time("frame", sequence=source_frame)
                rr.set_time("camera_time", timestamp=timestamps[source_frame].timestamp)
                rr.log(
                    "reference/left_camera",
                    rr.EncodedImage(
                        contents=encoded.tobytes(), media_type="image/jpeg"
                    ),
                )
                samples_written += 1
                next_sample_frame = samples_written * sample_period
                if samples_written % 250 == 0:
                    print(
                        f"Rerun camera samples: {samples_written}",
                        file=sys.stderr,
                        flush=True,
                    )
            source_frame += 1
    finally:
        capture.release()
    return {
        "video": str(video_path),
        "timestamps": str(timestamps_path),
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "samples_written": samples_written,
        "max_width": max_width,
        "jpeg_quality": jpeg_quality,
    }


def export_trajectory(
    trajectory_path: Path,
    output_path: Path,
    *,
    camera_video: Path | None = None,
    camera_timestamps: Path | None = None,
    camera_sample_fps: float = 2.0,
    camera_max_width: int = 960,
    jpeg_quality: int = 75,
    pose_sample_fps: float = 30.0,
) -> dict[str, object]:
    timestamps, frame_indices, transforms = load_trajectory(trajectory_path)
    if not len(timestamps):
        raise ValueError("Cannot export an empty trajectory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init("RoboEgo OpenVINS Trajectory", spawn=False)
    if camera_video is not None:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/reference/left_camera", name="Left camera"),
                rrb.Spatial3DView(origin="/world", name="VIO trajectory"),
                column_shares=[0.62, 0.38],
            ),
            auto_views=False,
            collapse_panels=True,
        )
    else:
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(origin="/world", name="VIO trajectory"),
            auto_views=False,
            collapse_panels=True,
        )
    rr.save(str(output_path), default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/trajectory_convention",
        rr.TextDocument(
            "T_world_camera maps points from the raw left-camera frame into the OpenVINS world frame."
        ),
        static=True,
    )
    positions = transforms[:, :3, 3].astype(np.float32)
    next_pose_timestamp = float(timestamps[0])
    logged_pose_count = 0
    for timestamp, frame_index, transform in zip(timestamps, frame_indices, transforms):
        if timestamp + 1e-9 < next_pose_timestamp and frame_index != frame_indices[-1]:
            continue
        rr.set_time("frame", sequence=int(frame_index))
        rr.set_time("camera_time", timestamp=float(timestamp))
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=transform[:3, 3],
                mat3x3=transform[:3, :3],
                axis_length=0.12,
            ),
        )
        rr.log(
            "world/current_camera_center",
            rr.Points3D([transform[:3, 3]], colors=[255, 190, 60], radii=0.018),
        )
        logged_pose_count += 1
        next_pose_timestamp = float(timestamps[0]) + logged_pose_count / pose_sample_fps

    rr.log(
        "world/trajectory",
        rr.LineStrips3D([positions], colors=[80, 210, 255], radii=0.01),
        static=True,
    )
    rr.log(
        "world/pose_samples",
        rr.Points3D(positions, colors=[255, 255, 255], radii=0.008),
        static=True,
    )
    camera_report = None
    if camera_video is not None and camera_timestamps is not None:
        camera_report = log_sampled_camera(
            camera_video,
            camera_timestamps,
            sample_fps=camera_sample_fps,
            max_width=camera_max_width,
            jpeg_quality=jpeg_quality,
        )

    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    recording = rr.get_global_data_recording()
    if recording is not None:
        recording.flush()
    rr.disconnect()

    report = {
        "schema_version": "1.0",
        "trajectory_npz": str(trajectory_path),
        "output_rrd": str(output_path),
        "pose_count": len(timestamps),
        "logged_pose_count": logged_pose_count,
        "pose_sample_fps": pose_sample_fps,
        "first_frame_index": int(frame_indices[0]),
        "last_frame_index": int(frame_indices[-1]),
        "coordinate_convention": "T_world_camera",
        "camera_reference": camera_report,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera_video", type=Path)
    parser.add_argument("--camera_timestamps", type=Path)
    parser.add_argument("--camera_sample_fps", type=float, default=2.0)
    parser.add_argument("--camera_max_width", type=int, default=960)
    parser.add_argument("--jpeg_quality", type=int, default=75)
    parser.add_argument("--pose_sample_fps", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if (args.camera_video is None) != (args.camera_timestamps is None):
        raise SystemExit("camera_video and camera_timestamps must be supplied together")
    if (
        args.camera_sample_fps <= 0
        or args.camera_max_width <= 0
        or args.pose_sample_fps <= 0
    ):
        raise SystemExit(
            "camera_sample_fps, camera_max_width, and pose_sample_fps must be positive"
        )
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("jpeg_quality must be between 1 and 100")
    report = export_trajectory(
        args.trajectory.expanduser().resolve(),
        args.output.expanduser().resolve(),
        camera_video=args.camera_video.expanduser().resolve()
        if args.camera_video
        else None,
        camera_timestamps=(
            args.camera_timestamps.expanduser().resolve()
            if args.camera_timestamps
            else None
        ),
        camera_sample_fps=args.camera_sample_fps,
        camera_max_width=args.camera_max_width,
        jpeg_quality=args.jpeg_quality,
        pose_sample_fps=args.pose_sample_fps,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
