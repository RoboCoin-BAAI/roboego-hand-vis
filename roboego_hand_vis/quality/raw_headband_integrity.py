#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CAMERA_STREAMS = ("left", "right", "bleft", "bright")
MASTER_STREAM = "left"
DEFAULT_VIDEO_EXT = ".mp4"
IMU_ALL_ZERO_EPS = 1e-12
IMU_FROZEN_MAX_CHANGE_RATE = 0.001
IMU_JUMP_MIN_ABS_DELTA = 5.0
IMU_JUMP_RELATIVE_RATIO = 50.0
IMU_IGNORED_SIGNAL_COLUMNS = frozenset(
    {
        "orientation_w",
        "orientation_x",
        "orientation_y",
        "orientation_z",
    }
)


class SessionIntegrityError(RuntimeError):
    """Raised when a source session cannot be quality checked safely."""


@dataclass(frozen=True)
class TimestampSeries:
    indices: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class SourceSession:
    session_dir: Path
    camera_streams: tuple[str, ...]
    timestamps_by_stream: dict[str, TimestampSeries]
    timestamp_duplicate_summaries: dict[str, dict]
    timestamp_alignment_summary: dict[str, dict]
    video_infos: dict[str, dict]
    width: int
    height: int
    fps: int
    start: int
    num_frames: int
    timestamp_tolerance_s: float
    video_ext: str


def normalize_video_ext(video_ext: str | None) -> str:
    if not video_ext:
        return DEFAULT_VIDEO_EXT
    video_ext = str(video_ext).strip()
    if not video_ext:
        return DEFAULT_VIDEO_EXT
    return video_ext if video_ext.startswith(".") else f".{video_ext}"


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve()


def default_quality_report_root(session_dir: str | Path) -> Path:
    return Path(session_dir).resolve()


def parse_timestamp_file(path: str | Path) -> TimestampSeries:
    path = Path(path)
    if not path.exists():
        raise SessionIntegrityError(f"Missing timestamp file: {path}")

    indices: list[int] = []
    timestamps: list[float] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SessionIntegrityError(f"Malformed timestamp line {line_no} in {path}: {raw_line!r}")
        try:
            indices.append(int(parts[0]))
            timestamps.append(float(parts[1]))
        except ValueError as exc:
            raise SessionIntegrityError(f"Malformed timestamp line {line_no} in {path}: {raw_line!r}") from exc

    if not timestamps:
        raise SessionIntegrityError(f"Empty timestamp file: {path}")

    index_arr = np.asarray(indices, dtype=np.int64)
    ts_arr = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(ts_arr) < 0):
        raise SessionIntegrityError(f"Timestamp file decreases: {path}")
    return TimestampSeries(indices=index_arr, timestamps=ts_arr)


def summarize_timestamp_duplicates(series: TimestampSeries) -> dict:
    if len(series.timestamps) <= 1:
        return {
            "frame_count": int(len(series.timestamps)),
            "duplicate_count": 0,
            "duplicate_rate": 0.0,
            "first_duplicate_index": None,
        }
    duplicate_mask = np.diff(series.timestamps) == 0
    duplicate_positions = np.where(duplicate_mask)[0] + 1
    duplicate_count = int(len(duplicate_positions))
    return {
        "frame_count": int(len(series.timestamps)),
        "duplicate_count": duplicate_count,
        "duplicate_rate": float(duplicate_count / float(len(series.timestamps) - 1)),
        "first_duplicate_index": int(series.indices[duplicate_positions[0]]) if duplicate_count else None,
    }


def assert_aligned_timestamps(
    timestamps_by_stream: dict[str, TimestampSeries],
    master_stream: str,
    start: int,
    num_frames: int,
    tolerance_s: float,
) -> dict[str, dict]:
    master = timestamps_by_stream[master_stream]
    end = start + num_frames
    if end > len(master.timestamps):
        raise SessionIntegrityError(f"Master timestamp stream {master_stream} has only {len(master.timestamps)} frames")

    master_slice = master.timestamps[start:end]
    summary: dict[str, dict] = {}
    for stream, series in timestamps_by_stream.items():
        if end > len(series.timestamps):
            raise SessionIntegrityError(f"Timestamp stream {stream} has only {len(series.timestamps)} frames")
        deltas = np.abs(series.timestamps[start:end] - master_slice)
        over = np.where(deltas > tolerance_s)[0]
        max_rel = int(np.argmax(deltas)) if len(deltas) else 0
        summary[stream] = {
            "frame_count": int(len(deltas)),
            "max_abs_delta_s": float(deltas[max_rel]) if len(deltas) else 0.0,
            "max_abs_delta_index": int(series.indices[start + max_rel]) if len(deltas) else None,
            "over_tolerance_count": int(len(over)),
            "over_tolerance_rate": float(len(over) / float(len(deltas))) if len(deltas) else 0.0,
            "first_over_tolerance_index": int(series.indices[start + int(over[0])]) if len(over) else None,
            "tolerance_s": float(tolerance_s),
        }
    return summary


def inspect_video(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise SessionIntegrityError(f"Missing video: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise SessionIntegrityError(f"Cannot open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if width <= 0 or height <= 0:
            raise SessionIntegrityError(f"Video has invalid dimensions: {path}")
        return {"width": width, "height": height, "frame_count": frame_count, "fps": fps}
    finally:
        cap.release()


def inspect_imu(path: str | Path) -> dict | None:
    path = Path(path).resolve()
    if not path.exists():
        return None

    timestamps: list[float] = []
    numeric_values_by_column: dict[str, list[float]] = {}
    candidate_columns: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "timestamp" not in reader.fieldnames:
            raise SessionIntegrityError(f"IMU file is missing timestamp column: {path}")
        ignored_columns = [field for field in reader.fieldnames if field in IMU_IGNORED_SIGNAL_COLUMNS]
        candidate_columns = [
            field
            for field in reader.fieldnames
            if field != "timestamp" and field not in IMU_IGNORED_SIGNAL_COLUMNS
        ]
        numeric_values_by_column = {field: [] for field in candidate_columns}
        for row_no, row in enumerate(reader, start=2):
            value = row.get("timestamp")
            try:
                timestamps.append(float(value))
            except (TypeError, ValueError) as exc:
                raise SessionIntegrityError(f"Malformed IMU timestamp on line {row_no} in {path}: {value!r}") from exc
            for column in candidate_columns:
                if column not in numeric_values_by_column:
                    continue
                raw_value = row.get(column)
                try:
                    numeric_values_by_column[column].append(float(raw_value))
                except (TypeError, ValueError):
                    numeric_values_by_column.pop(column, None)

    if not timestamps:
        raise SessionIntegrityError(f"Empty IMU file: {path}")

    ts_arr = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(ts_arr) < 0):
        raise SessionIntegrityError(f"IMU timestamp decreases: {path}")

    duplicate_mask = np.diff(ts_arr) == 0
    duplicate_positions = np.where(duplicate_mask)[0] + 1
    nonzero_deltas = np.diff(ts_arr)[np.diff(ts_arr) > 0]
    estimated_fps = float(1.0 / np.median(nonzero_deltas)) if len(nonzero_deltas) else None
    return {
        "path": str(path),
        "sample_count": int(len(ts_arr)),
        "first_timestamp": float(ts_arr[0]),
        "last_timestamp": float(ts_arr[-1]),
        "duration_s": float(ts_arr[-1] - ts_arr[0]) if len(ts_arr) > 1 else 0.0,
        "estimated_fps": estimated_fps,
        "duplicate_count": int(len(duplicate_positions)),
        "duplicate_rate": float(len(duplicate_positions) / float(len(ts_arr) - 1)) if len(ts_arr) > 1 else 0.0,
        "first_duplicate_index": int(duplicate_positions[0]) if len(duplicate_positions) else None,
        "signal_quality": summarize_imu_signal_quality(numeric_values_by_column, ignored_columns=ignored_columns),
    }


def summarize_imu_signal_quality(values_by_column: dict[str, list[float]], ignored_columns: list[str] | None = None) -> dict:
    columns: dict[str, dict] = {}
    all_zero_columns: list[str] = []
    frozen_columns: list[str] = []
    jump_columns: list[str] = []
    total_values = 0
    finite_values = 0

    for column, values in sorted(values_by_column.items()):
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        total_values += int(len(arr))
        finite_values += int(np.count_nonzero(finite_mask))
        finite = arr[finite_mask]
        if len(finite) == 0:
            columns[column] = {
                "sample_count": int(len(arr)),
                "finite_rate": 0.0,
                "all_zero": False,
                "frozen": False,
                "jump": False,
                "change_rate": 0.0,
                "max_abs_delta": None,
                "max_jump_sample_index": None,
            }
            continue

        all_zero = bool(np.all(np.abs(finite) <= IMU_ALL_ZERO_EPS))
        diffs = np.abs(np.diff(finite))
        changed = diffs > IMU_ALL_ZERO_EPS
        change_rate = float(np.count_nonzero(changed) / float(len(diffs))) if len(diffs) else 0.0
        frozen = bool(len(finite) > 1 and change_rate <= IMU_FROZEN_MAX_CHANGE_RATE)
        max_abs_delta = float(np.max(diffs)) if len(diffs) else 0.0
        max_jump_sample_index = int(np.argmax(diffs) + 1) if len(diffs) else None
        positive_diffs = diffs[diffs > IMU_ALL_ZERO_EPS]
        typical_delta = float(np.median(positive_diffs)) if len(positive_diffs) else 0.0
        if typical_delta > IMU_ALL_ZERO_EPS:
            jump = bool(max_abs_delta >= IMU_JUMP_MIN_ABS_DELTA and max_abs_delta / typical_delta >= IMU_JUMP_RELATIVE_RATIO)
        else:
            jump = bool(max_abs_delta >= IMU_JUMP_MIN_ABS_DELTA)

        if all_zero:
            all_zero_columns.append(column)
        if frozen:
            frozen_columns.append(column)
        if jump:
            jump_columns.append(column)
        columns[column] = {
            "sample_count": int(len(arr)),
            "finite_rate": float(np.count_nonzero(finite_mask) / float(len(arr))),
            "all_zero": all_zero,
            "frozen": frozen,
            "jump": jump,
            "change_rate": change_rate,
            "max_abs_delta": max_abs_delta,
            "max_jump_sample_index": max_jump_sample_index,
            "typical_abs_delta": typical_delta,
        }

    return {
        "numeric_column_count": len(columns),
        "finite_rate": float(finite_values / float(total_values)) if total_values else None,
        "all_zero_columns": all_zero_columns,
        "frozen_columns": frozen_columns,
        "jump_columns": jump_columns,
        "ignored_columns": sorted(ignored_columns or []),
        "columns": columns,
        "thresholds": {
            "all_zero_eps": IMU_ALL_ZERO_EPS,
            "frozen_max_change_rate": IMU_FROZEN_MAX_CHANGE_RATE,
            "jump_min_abs_delta": IMU_JUMP_MIN_ABS_DELTA,
            "jump_relative_ratio": IMU_JUMP_RELATIVE_RATIO,
        },
    }


def validate_source_session(
    session_dir: str | Path,
    camera_streams: tuple[str, ...],
    start: int,
    num_frames: int | None,
    timestamp_tolerance_s: float | None,
    video_ext: str = DEFAULT_VIDEO_EXT,
    video_inspector=inspect_video,
) -> SourceSession:
    session_dir = Path(session_dir).resolve()
    video_ext = normalize_video_ext(video_ext)
    if not session_dir.exists():
        raise SessionIntegrityError(f"Missing session directory: {session_dir}")

    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        raise SessionIntegrityError(f"Missing meta.json: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionIntegrityError(f"Malformed meta.json: {meta_path}") from exc

    fps = int(round(float(meta.get("fps", 30))))
    if fps <= 0:
        raise SessionIntegrityError(f"Invalid fps in meta.json: {fps}")
    if timestamp_tolerance_s is None:
        timestamp_tolerance_s = (1.0 / float(fps)) + 0.001

    timestamps_by_stream = {
        stream: parse_timestamp_file(session_dir / "timestamps" / f"{stream}_timestamps.txt")
        for stream in camera_streams
    }
    if MASTER_STREAM not in timestamps_by_stream:
        raise SessionIntegrityError(f"Master stream {MASTER_STREAM!r} must be enabled")

    video_infos = {
        stream: video_inspector(session_dir / "videos" / f"{stream}{video_ext}")
        for stream in camera_streams
    }
    timestamp_duplicate_summaries = {
        stream: summarize_timestamp_duplicates(series)
        for stream, series in timestamps_by_stream.items()
    }

    first_info = video_infos[camera_streams[0]]
    width = int(first_info["width"])
    height = int(first_info["height"])
    for stream, info in video_infos.items():
        if int(info["width"]) != width or int(info["height"]) != height:
            raise SessionIntegrityError(f"Video dimensions for {stream} differ from {camera_streams[0]}")

    available_frames = min(
        [len(series.timestamps) for series in timestamps_by_stream.values()]
        + [int(info["frame_count"]) for info in video_infos.values()]
    )
    if start < 0:
        raise SessionIntegrityError(f"start must be non-negative, got {start}")
    if num_frames is None:
        num_frames = available_frames - start
    if num_frames <= 0:
        raise SessionIntegrityError(f"num_frames must be positive, got {num_frames}")

    end = start + num_frames
    if end > available_frames:
        raise SessionIntegrityError(f"Requested frames [{start}, {end}) exceed available frame count {available_frames}")
    for stream, info in video_infos.items():
        if int(info["frame_count"]) < end:
            raise SessionIntegrityError(f"Video frame count is lower than requested frame count for {stream}")

    timestamp_alignment_summary = assert_aligned_timestamps(
        timestamps_by_stream,
        MASTER_STREAM,
        start,
        num_frames,
        float(timestamp_tolerance_s),
    )
    return SourceSession(
        session_dir=session_dir,
        camera_streams=tuple(camera_streams),
        timestamps_by_stream=timestamps_by_stream,
        timestamp_duplicate_summaries=timestamp_duplicate_summaries,
        timestamp_alignment_summary=timestamp_alignment_summary,
        video_infos=video_infos,
        width=width,
        height=height,
        fps=fps,
        start=start,
        num_frames=num_frames,
        timestamp_tolerance_s=float(timestamp_tolerance_s),
        video_ext=video_ext,
    )


def build_quality_report(session: SourceSession, environment_name: str) -> dict:
    imu_summary = inspect_imu(session.session_dir / "imu" / "imu0.csv")
    if imu_summary is not None:
        imu_start = float(imu_summary["first_timestamp"])
        imu_end = float(imu_summary["last_timestamp"])
        last_frame_idx = session.start + session.num_frames - 1
        imu_summary["start_delta_to_cameras_s"] = {
            stream: float(imu_start - session.timestamps_by_stream[stream].timestamps[session.start])
            for stream in session.camera_streams
        }
        imu_summary["end_delta_to_cameras_s"] = {
            stream: float(imu_end - session.timestamps_by_stream[stream].timestamps[last_frame_idx])
            for stream in session.camera_streams
        }

    return {
        "mode": "quality_report",
        "source_session": str(session.session_dir),
        "environment": environment_name,
        "report_time_utc": datetime.now(timezone.utc).isoformat(),
        "camera_streams": list(session.camera_streams),
        "frame_window": {
            "start": int(session.start),
            "num_frames": int(session.num_frames),
        },
        "fps": int(session.fps),
        "image_size": {
            "width": int(session.width),
            "height": int(session.height),
        },
        "videos": {
            stream: str(session.session_dir / "videos" / f"{stream}{session.video_ext}")
            for stream in session.camera_streams
        },
        "timestamps": {
            stream: str(session.session_dir / "timestamps" / f"{stream}_timestamps.txt")
            for stream in session.camera_streams
        },
        "video_infos": session.video_infos,
        "timestamp_duplicates": session.timestamp_duplicate_summaries,
        "timestamp_alignment": session.timestamp_alignment_summary,
        "imu": imu_summary,
        "timestamp_tolerance_s": float(session.timestamp_tolerance_s),
        "notes": [
            "Video check reads container metadata only; it does not decode all frames.",
            "Alignment statistics are recorded for review and do not fail this report unless source integrity checks fail.",
            "IMU is checked for timestamp quality only; it is not converted into state.",
        ],
    }


def export_quality_report(
    session_dir: str | Path,
    output_root: str | Path,
    camera_streams: tuple[str, ...] = DEFAULT_CAMERA_STREAMS,
    start: int = 0,
    num_frames: int | None = None,
    timestamp_tolerance_s: float | None = None,
    video_ext: str = DEFAULT_VIDEO_EXT,
    overwrite: bool = False,
    environment_name: str = "py312",
    video_inspector=inspect_video,
) -> Path:
    session = validate_source_session(
        session_dir,
        tuple(camera_streams),
        start=start,
        num_frames=num_frames,
        timestamp_tolerance_s=timestamp_tolerance_s,
        video_ext=video_ext,
        video_inspector=video_inspector,
    )
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    annotation_dir = output_root / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    report_path = annotation_dir / "quality_report.json"
    if report_path.exists() and not overwrite:
        raise SessionIntegrityError(f"Quality report exists; pass --overwrite to replace it: {report_path}")
    report_path.write_text(
        json.dumps(build_quality_report(session, environment_name), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check a raw RoboEgo headband session and write a quality report.")
    parser.add_argument("--session", required=True, help="Raw headband session directory.")
    parser.add_argument("--output_root", default=None, help="Report root. Defaults to the source session directory.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--timestamp_tolerance_s", type=float, default=None)
    parser.add_argument("--video_ext", default=DEFAULT_VIDEO_EXT, help="Raw camera video extension, e.g. .mp4 or .mkv.")
    parser.add_argument("--camera_stream", action="append", choices=DEFAULT_CAMERA_STREAMS, default=list(DEFAULT_CAMERA_STREAMS))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cwd = Path.cwd()
    session_dir = resolve_path(args.session, cwd)
    output_root = resolve_path(args.output_root, cwd) if args.output_root else default_quality_report_root(session_dir)
    report_path = export_quality_report(
        session_dir=session_dir,
        output_root=output_root,
        camera_streams=tuple(args.camera_stream),
        start=args.start,
        num_frames=args.num_frames,
        timestamp_tolerance_s=args.timestamp_tolerance_s,
        video_ext=args.video_ext,
        overwrite=args.overwrite,
    )
    print(json.dumps({"report_path": str(report_path), "mode": "quality_report"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
