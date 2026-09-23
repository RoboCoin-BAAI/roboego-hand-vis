#!/usr/bin/env python3
"""Stream a raw camera view and OpenVINS top-down trajectory into an H.264 MP4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

STAGE_DIR = Path(__file__).resolve().parents[2] / "roboego_hand_vis" / "vio"
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from preview import fit_topdown_projection


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    scale: float = 0.52,
    color: tuple[int, int, int] = (232, 232, 232),
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as trajectory:
        frame_indices = np.asarray(trajectory["frame_index"], dtype=np.int64)
        timestamps = np.asarray(trajectory["timestamp_camera"], dtype=np.float64)
        transforms = np.asarray(trajectory["T_world_camera"], dtype=np.float64)
    if (
        transforms.shape != (len(frame_indices), 4, 4)
        or timestamps.shape != frame_indices.shape
    ):
        raise ValueError("Trajectory arrays are not aligned")
    if not len(frame_indices) or not np.all(np.diff(frame_indices) > 0):
        raise ValueError(
            "Trajectory frame indices must be non-empty and strictly increasing"
        )
    return frame_indices, timestamps, transforms


def start_encoder(
    output: Path, width: int, height: int, fps: float, crf: int
) -> subprocess.Popen[bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def run(args: argparse.Namespace) -> dict[str, object]:
    video_path = args.video.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    frame_indices, timestamps, transforms = load_trajectory(trajectory_path)
    positions = transforms[:, :3, 3]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or source_frame_count <= 0:
        raise ValueError("Source video has invalid FPS or frame count")
    start_frame = max(0, round(args.start_s * source_fps))
    requested_end = (
        start_frame + round(args.duration_s * source_fps)
        if args.duration_s is not None
        else source_frame_count
    )
    end_frame = min(source_frame_count, requested_end)
    if start_frame >= end_frame:
        raise ValueError("Selected preview interval is empty")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    camera_width = round(args.width * args.camera_fraction)
    camera_width -= camera_width % 2
    plot_width = args.width - camera_width
    projection = fit_topdown_projection(
        positions[:, :2], width=plot_width, height=args.height
    )
    pose_pixels = projection.project(positions[:, :2])

    base_plot = np.full((args.height, plot_width, 3), (24, 26, 29), dtype=np.uint8)
    for fraction in np.linspace(0.1, 0.9, 5):
        x = round(plot_width * fraction)
        y = round(args.height * fraction)
        cv2.line(base_plot, (x, 28), (x, args.height - 28), (48, 51, 56), 1)
        cv2.line(base_plot, (28, y), (plot_width - 28, y), (48, 51, 56), 1)
    cv2.polylines(base_plot, [pose_pixels], False, (75, 80, 86), 1, cv2.LINE_AA)
    draw_text(
        base_plot, "OpenVINS top view", (18, 26), scale=0.56, color=(255, 214, 120)
    )
    draw_text(
        base_plot,
        f"scale: {projection.pixels_per_meter:.1f} px/m",
        (18, args.height - 18),
        scale=0.43,
        color=(175, 180, 188),
    )
    trail_plot = base_plot.copy()
    last_trail_pose = 0

    encoder = start_encoder(output_path, args.width, args.height, args.fps, args.crf)
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg stdin is unavailable")
    output_period_frames = source_fps / args.fps
    next_output_source_frame = float(start_frame)
    frames_written = 0
    source_frame = start_frame
    try:
        while source_frame < end_frame:
            ok = capture.grab()
            if not ok:
                break
            if source_frame + 1e-6 >= next_output_source_frame:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not decode source frame {source_frame}")
                pose_index = int(
                    np.searchsorted(frame_indices, source_frame, side="right") - 1
                )
                camera_panel = letterbox(frame, camera_width, args.height)
                draw_text(
                    camera_panel,
                    f"left camera  frame {source_frame}",
                    (18, 28),
                    scale=0.58,
                )
                draw_text(
                    camera_panel,
                    f"session time {source_frame / source_fps:.2f} s",
                    (18, 54),
                    scale=0.5,
                )

                plot_panel = trail_plot.copy()
                if pose_index >= 0:
                    for index in range(last_trail_pose + 1, pose_index + 1):
                        cv2.line(
                            trail_plot,
                            tuple(pose_pixels[index - 1]),
                            tuple(pose_pixels[index]),
                            (255, 190, 70),
                            2,
                            cv2.LINE_AA,
                        )
                    last_trail_pose = max(last_trail_pose, pose_index)
                    plot_panel = trail_plot.copy()
                    current = tuple(pose_pixels[pose_index])
                    cv2.circle(plot_panel, current, 6, (80, 235, 255), -1, cv2.LINE_AA)
                    forward_xy = transforms[pose_index, :2, 2]
                    forward_norm = float(np.linalg.norm(forward_xy))
                    if forward_norm > 1e-6:
                        endpoint = np.rint(
                            pose_pixels[pose_index] + forward_xy / forward_norm * 28.0
                        ).astype(np.int32)
                        cv2.arrowedLine(
                            plot_panel,
                            current,
                            tuple(endpoint),
                            (90, 255, 150),
                            2,
                            cv2.LINE_AA,
                            tipLength=0.3,
                        )
                    elapsed = timestamps[pose_index] - timestamps[0]
                    draw_text(
                        plot_panel,
                        f"VIO: tracking  t={elapsed:.2f}s",
                        (18, 52),
                        scale=0.48,
                    )
                    draw_text(
                        plot_panel,
                        f"x {positions[pose_index, 0]:.2f}  y {positions[pose_index, 1]:.2f} m",
                        (18, 76),
                        scale=0.46,
                    )
                else:
                    draw_text(
                        plot_panel,
                        "VIO: initializing",
                        (18, 52),
                        scale=0.5,
                        color=(120, 190, 255),
                    )

                composite = np.concatenate((camera_panel, plot_panel), axis=1)
                encoder.stdin.write(composite.tobytes())
                frames_written += 1
                next_output_source_frame = (
                    start_frame + frames_written * output_period_frames
                )
                if frames_written % 500 == 0:
                    print(
                        f"preview frames: {frames_written}", file=sys.stderr, flush=True
                    )
            source_frame += 1
    finally:
        capture.release()
        encoder.stdin.close()
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")

    report = {
        "schema_version": "1.0",
        "video": str(video_path),
        "trajectory": str(trajectory_path),
        "output_mp4": str(output_path),
        "camera": "left",
        "frames_written": frames_written,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "streaming_export": True,
        "rrd_written": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--camera_fraction", type=float, default=0.70)
    parser.add_argument("--start_s", type=float, default=0.0)
    parser.add_argument("--duration_s", type=float)
    parser.add_argument("--crf", type=int, default=23)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("fps, width, and height must be positive")
    if not 0.4 <= args.camera_fraction <= 0.85:
        raise SystemExit("camera_fraction must be between 0.4 and 0.85")
    if args.width % 2 or args.height % 2:
        raise SystemExit("width and height must be even for yuv420p output")
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
