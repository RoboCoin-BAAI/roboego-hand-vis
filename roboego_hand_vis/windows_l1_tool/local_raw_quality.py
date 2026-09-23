#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def find_named_video(video_dir: str | Path, name: str, preferred_ext: str = ".mp4") -> Path:
    video_dir = Path(video_dir)
    preferred = video_dir / f"{name}{preferred_ext}"
    if preferred.exists():
        return preferred
    matches = sorted(video_dir.glob(f"{name}.*"))
    if not matches:
        raise FileNotFoundError(f"Missing video for stream {name}: {preferred}")
    if len(matches) > 1:
        raise FileExistsError(f"Multiple videos for stream {name}: {matches}")
    return matches[0]


DEFAULT_CAMERA_STREAMS = ("left", "right", "bleft", "bright")
MASTER_STREAM = "left"
DEFAULT_VIDEO_EXT = ".mp4"
DEFAULT_AUDIO_RELATIVE_PATH = Path("audio") / "audio.wav"
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
    video_paths: dict[str, Path]
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
            "max_consecutive_duplicate_frames": 0,
            "first_max_consecutive_duplicate_index": None,
        }
    duplicate_mask = np.diff(series.timestamps) == 0
    duplicate_positions = np.where(duplicate_mask)[0] + 1
    duplicate_count = int(len(duplicate_positions))
    max_run_length, first_max_run_index = _longest_equal_timestamp_run(series)
    return {
        "frame_count": int(len(series.timestamps)),
        "duplicate_count": duplicate_count,
        "duplicate_rate": float(duplicate_count / float(len(series.timestamps) - 1)),
        "first_duplicate_index": int(series.indices[duplicate_positions[0]]) if duplicate_count else None,
        "max_consecutive_duplicate_frames": max_run_length,
        "first_max_consecutive_duplicate_index": first_max_run_index,
    }


def _longest_equal_timestamp_run(series: TimestampSeries) -> tuple[int, int | None]:
    timestamps = series.timestamps
    if len(timestamps) <= 1:
        return 0, None
    max_run_length = 0
    max_run_start = None
    run_start = 0
    for idx in range(1, len(timestamps)):
        if timestamps[idx] == timestamps[idx - 1]:
            continue
        run_length = idx - run_start
        if run_length > 1 and run_length > max_run_length:
            max_run_length = run_length
            max_run_start = run_start
        run_start = idx
    run_length = len(timestamps) - run_start
    if run_length > 1 and run_length > max_run_length:
        max_run_length = run_length
        max_run_start = run_start
    if max_run_start is None:
        return 0, None
    return int(max_run_length), int(series.indices[max_run_start])


def repair_duplicate_timestamps(series: TimestampSeries, fps: float) -> tuple[TimestampSeries, dict]:
    """Return an in-memory timestamp series with equal-timestamp runs interpolated."""
    timestamps = np.asarray(series.timestamps, dtype=np.float64).copy()
    duplicate_summary = summarize_timestamp_duplicates(series)
    if int(duplicate_summary["duplicate_count"]) == 0:
        return series, {"applied": False, **duplicate_summary}

    nominal_step = 1.0 / float(fps) if fps > 0 else None
    idx = 0
    repaired_runs: list[dict[str, int]] = []
    while idx < len(timestamps):
        run_end = idx + 1
        while run_end < len(timestamps) and timestamps[run_end] == timestamps[idx]:
            run_end += 1
        run_length = run_end - idx
        if run_length > 1:
            repaired_runs.append(
                {
                    "start_index": int(series.indices[idx]),
                    "end_index": int(series.indices[run_end - 1]),
                    "frame_count": int(run_length),
                }
            )
            prev_value = timestamps[idx - 1] if idx > 0 else None
            next_value = timestamps[run_end] if run_end < len(timestamps) else None
            if prev_value is not None and next_value is not None and next_value > prev_value:
                step = float((next_value - prev_value) / float(run_length + 1))
                for offset in range(run_length):
                    timestamps[idx + offset] = float(prev_value + step * float(offset + 1))
            elif prev_value is not None and nominal_step is not None:
                for offset in range(run_length):
                    timestamps[idx + offset] = float(prev_value + nominal_step * float(offset + 1))
            elif next_value is not None and nominal_step is not None:
                for offset in range(run_length):
                    timestamps[idx + offset] = float(next_value - nominal_step * float(run_length - offset))
        idx = run_end

    return (
        TimestampSeries(indices=series.indices.copy(), timestamps=timestamps),
        {
            "applied": True,
            "method": "interpolate_equal_timestamp_runs",
            **duplicate_summary,
            "runs": repaired_runs,
        },
    )


def repair_duplicate_timestamps_from_reference(
    series: TimestampSeries,
    reference: TimestampSeries | None,
    fps: float,
) -> tuple[TimestampSeries, dict]:
    if reference is None or len(reference.timestamps) != len(series.timestamps):
        return repair_duplicate_timestamps(series, fps)
    repaired, summary = repair_duplicate_timestamps(series, fps)
    if not summary.get("applied"):
        return repaired, summary

    timestamps = np.asarray(series.timestamps, dtype=np.float64).copy()
    for run in summary.get("runs", []):
        start_index = int(run["start_index"])
        end_index = int(run["end_index"])
        positions = np.where((series.indices >= start_index) & (series.indices <= end_index))[0]
        if len(positions) == 0:
            continue
        reference_values = reference.timestamps[positions]
        if len(reference_values) > 1 and not np.all(np.diff(reference_values) > 0):
            continue
        timestamps[positions] = reference_values

    summary = dict(summary)
    summary["method"] = "copy_master_timestamps_for_duplicate_runs"
    return TimestampSeries(indices=series.indices.copy(), timestamps=timestamps), summary


def repair_duplicate_timestamps_by_stream(
    timestamps_by_stream: dict[str, TimestampSeries],
    fps: float,
    master_stream: str = MASTER_STREAM,
) -> tuple[dict[str, TimestampSeries], dict]:
    repaired: dict[str, TimestampSeries] = {}
    streams: dict[str, dict] = {}
    master_series = timestamps_by_stream.get(master_stream)
    for stream, series in timestamps_by_stream.items():
        reference = None if stream == master_stream else master_series
        repaired_series, summary = repair_duplicate_timestamps_from_reference(series, reference, fps)
        repaired[stream] = repaired_series
        if summary.get("applied"):
            streams[stream] = summary
    return repaired, {
        "applied": bool(streams),
        "method": "interpolate_equal_timestamp_runs",
        "streams": streams,
    }


def write_timestamp_file(path: str | Path, series: TimestampSeries) -> None:
    path = Path(path)
    lines = [f"{int(idx)} {float(timestamp):.9f}" for idx, timestamp in zip(series.indices, series.timestamps)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_repair_timestamp_files(
    *,
    timestamp_paths: dict[str, Path],
    timestamp_duplicate_summaries: dict[str, dict],
    repaired_timestamps_by_stream: dict[str, TimestampSeries],
    timestamp_alignment_repaired_summary: dict[str, dict],
    repair_timestamp_files: bool,
    max_repair_duplicate_rate: float,
    max_repair_consecutive_duplicate_frames: int,
) -> dict:
    result = {
        "enabled": bool(repair_timestamp_files),
        "applied": False,
        "streams": {},
        "skipped_streams": {},
    }
    if not repair_timestamp_files:
        return result

    for stream, summary in sorted(timestamp_duplicate_summaries.items()):
        if int(summary.get("duplicate_count", 0)) <= 0:
            continue
        reasons = []
        if float(summary.get("duplicate_rate", 0.0)) > float(max_repair_duplicate_rate):
            reasons.append("duplicate_rate_over_limit")
        if int(summary.get("max_consecutive_duplicate_frames", 0)) > int(max_repair_consecutive_duplicate_frames):
            reasons.append("consecutive_duplicate_frames_over_limit")
        repaired_alignment = timestamp_alignment_repaired_summary.get(stream) or {}
        if reasons:
            result["skipped_streams"][stream] = {
                "reasons": reasons,
                "duplicate_summary": summary,
                "repaired_alignment": repaired_alignment,
            }
            continue
        path = timestamp_paths.get(stream)
        repaired_series = repaired_timestamps_by_stream.get(stream)
        if path is None or repaired_series is None:
            result["skipped_streams"][stream] = {
                "reasons": ["missing_timestamp_path_or_repaired_series"],
                "duplicate_summary": summary,
            }
            continue
        write_timestamp_file(path, repaired_series)
        result["applied"] = True
        result["streams"][stream] = {
            "written": True,
            "path": str(path),
            "duplicate_summary": summary,
            "repaired_alignment": repaired_alignment,
        }
    return result


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
        frame_readable, _frame = cap.read()
        return {
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": fps,
            "frame_readable": bool(frame_readable),
        }
    finally:
        cap.release()


def inspect_audio(path: str | Path) -> dict | None:
    path = Path(path).resolve()
    if not path.exists():
        return None

    with wave.open(str(path), "rb") as handle:
        frame_count = int(handle.getnframes())
        sample_rate = int(handle.getframerate())
        channel_count = int(handle.getnchannels())
        sample_width_bytes = int(handle.getsampwidth())
    if sample_rate <= 0:
        raise SessionIntegrityError(f"Audio has invalid sample rate: {path}")
    return {
        "path": str(path),
        "exists": True,
        "readable": True,
        "frame_count": frame_count,
        "sample_rate": sample_rate,
        "channel_count": channel_count,
        "sample_width_bytes": sample_width_bytes,
        "duration_s": float(frame_count / float(sample_rate)),
    }


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

    try:
        video_paths = {
            stream: find_named_video(session_dir / "videos", stream, preferred_ext=video_ext)
            for stream in camera_streams
        }
    except (FileNotFoundError, FileExistsError) as exc:
        raise SessionIntegrityError(str(exc)) from exc
    video_infos = {stream: video_inspector(path) for stream, path in video_paths.items()}
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
        video_paths=video_paths,
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
            stream: str(session.video_paths[stream])
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


def _read_meta_for_report(meta_path: Path) -> tuple[dict, int, list[str]]:
    errors: list[str] = []
    if not meta_path.exists():
        return {}, 30, [f"Missing meta.json: {meta_path}"]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, 30, [f"Malformed meta.json: {meta_path}: {exc}"]
    try:
        fps = int(round(float(meta.get("fps", 30))))
    except (TypeError, ValueError):
        errors.append(f"Invalid fps in meta.json: {meta.get('fps')!r}")
        fps = 30
    if fps <= 0:
        errors.append(f"Invalid fps in meta.json: {fps}")
        fps = 30
    return meta, fps, errors


def _presence(path: Path, *, required: bool = True) -> dict:
    return {"path": str(path), "exists": path.exists(), "required": required}


def _count_delta(counts: dict[str, int]) -> int:
    values = [int(value) for value in counts.values()]
    return int(max(values) - min(values)) if values else 0


def _build_frame_count_consistency(
    *,
    camera_streams: tuple[str, ...],
    video_infos: dict[str, dict],
    timestamps_by_stream: dict[str, TimestampSeries],
    available_frames: int,
) -> dict:
    video_counts = {
        stream: int(video_infos[stream]["frame_count"])
        for stream in camera_streams
        if stream in video_infos and video_infos[stream].get("readable", True) and video_infos[stream].get("frame_count") is not None
    }
    timestamp_counts = {
        stream: int(len(timestamps_by_stream[stream].timestamps))
        for stream in camera_streams
        if stream in timestamps_by_stream
    }
    video_timestamp_deltas = {
        stream: abs(int(video_counts[stream]) - int(timestamp_counts[stream]))
        for stream in camera_streams
        if stream in video_counts and stream in timestamp_counts
    }
    return {
        "video_frame_counts": video_counts,
        "timestamp_frame_counts": timestamp_counts,
        "video_timestamp_frame_count_deltas": video_timestamp_deltas,
        "max_video_frame_count_delta": _count_delta(video_counts),
        "max_timestamp_frame_count_delta": _count_delta(timestamp_counts),
        "max_video_timestamp_count_delta": max(video_timestamp_deltas.values()) if video_timestamp_deltas else 0,
        "available_frames": int(available_frames),
        "policy": "trim_to_min_available_frames",
    }


def _build_imu_time_coverage(
    *,
    imu_summary: dict | None,
    timestamps_by_stream: dict[str, TimestampSeries],
    start: int,
    num_frames: int,
    fps: int,
) -> dict:
    if not isinstance(imu_summary, dict) or imu_summary.get("first_timestamp") is None:
        return {"computed": False, "reason": "imu_missing_or_unreadable"}
    if num_frames <= 0 or not timestamps_by_stream:
        return {"computed": False, "reason": "no_camera_frame_window"}

    imu_start = float(imu_summary["first_timestamp"])
    imu_end = float(imu_summary["last_timestamp"])
    last_frame_idx = int(start) + int(num_frames) - 1
    streams: dict[str, dict] = {}
    max_missing_gap_s = 0.0
    for stream, series in sorted(timestamps_by_stream.items()):
        if int(start) >= len(series.timestamps) or last_frame_idx >= len(series.timestamps):
            continue
        camera_start = float(series.timestamps[int(start)])
        camera_end = float(series.timestamps[last_frame_idx])
        start_missing_gap_s = max(0.0, imu_start - camera_start)
        end_missing_gap_s = max(0.0, camera_end - imu_end)
        max_missing_gap_s = max(max_missing_gap_s, start_missing_gap_s, end_missing_gap_s)
        streams[stream] = {
            "camera_start_timestamp": camera_start,
            "camera_end_timestamp": camera_end,
            "imu_start_delta_to_camera_s": float(imu_start - camera_start),
            "imu_end_delta_to_camera_s": float(imu_end - camera_end),
            "start_missing_gap_s": start_missing_gap_s,
            "end_missing_gap_s": end_missing_gap_s,
            "start_missing_gap_frames": float(start_missing_gap_s * float(fps)),
            "end_missing_gap_frames": float(end_missing_gap_s * float(fps)),
            "covered": start_missing_gap_s == 0.0 and end_missing_gap_s == 0.0,
        }

    return {
        "computed": bool(streams),
        "fps": int(fps),
        "imu_start_timestamp": imu_start,
        "imu_end_timestamp": imu_end,
        "max_missing_gap_s": max_missing_gap_s,
        "max_missing_gap_frames": float(max_missing_gap_s * float(fps)),
        "streams": streams,
    }


def _build_audio_video_duration_consistency(
    *,
    audio_summary: dict | None,
    video_infos: dict[str, dict],
) -> dict:
    if not isinstance(audio_summary, dict) or not audio_summary.get("readable", True):
        return {"computed": False, "reason": "audio_missing_or_unreadable"}
    audio_duration_s = audio_summary.get("duration_s")
    if audio_duration_s is None:
        return {"computed": False, "reason": "audio_duration_missing"}

    streams: dict[str, dict] = {}
    max_abs_delta_s = 0.0
    for stream, info in sorted(video_infos.items()):
        if not info.get("readable", True):
            continue
        frame_count = info.get("frame_count")
        fps = info.get("fps")
        if frame_count is None or fps is None or float(fps) <= 0:
            continue
        video_duration_s = float(frame_count) / float(fps)
        delta_s = float(audio_duration_s) - video_duration_s
        max_abs_delta_s = max(max_abs_delta_s, abs(delta_s))
        streams[stream] = {
            "video_duration_s": video_duration_s,
            "audio_duration_s": float(audio_duration_s),
            "delta_s": delta_s,
            "abs_delta_s": abs(delta_s),
        }

    if not streams:
        return {"computed": False, "reason": "no_readable_video_duration"}
    return {
        "computed": True,
        "audio_duration_s": float(audio_duration_s),
        "max_abs_delta_s": max_abs_delta_s,
        "streams": streams,
    }


def build_resilient_quality_report(
    session_dir: str | Path,
    camera_streams: tuple[str, ...],
    start: int,
    num_frames: int | None,
    timestamp_tolerance_s: float | None,
    video_ext: str,
    environment_name: str,
    repair_timestamp_files: bool = False,
    max_repair_duplicate_rate: float = 0.001,
    max_repair_consecutive_duplicate_frames: int = 5,
    video_inspector=inspect_video,
) -> dict:
    """Build a raw quality report without raising for bad source data."""
    session_dir = Path(session_dir).resolve()
    video_ext = normalize_video_ext(video_ext)
    camera_streams = tuple(camera_streams)
    meta_path = session_dir / "meta.json"
    imu_path = session_dir / "imu" / "imu0.csv"
    audio_path = session_dir / DEFAULT_AUDIO_RELATIVE_PATH
    meta, fps, meta_errors = _read_meta_for_report(meta_path)
    if timestamp_tolerance_s is None:
        timestamp_tolerance_s = (1.0 / float(fps)) + 0.001

    collection_errors: list[str] = list(meta_errors)
    modality_presence: dict[str, dict] = {
        "session_dir": {"path": str(session_dir), "exists": session_dir.exists(), "required": True},
        "meta": _presence(meta_path, required=True),
        "imu": _presence(imu_path, required=False),
        "audio": _presence(audio_path, required=True),
    }
    videos: dict[str, str] = {}
    timestamps: dict[str, str] = {}
    timestamp_paths: dict[str, Path] = {}
    video_infos: dict[str, dict] = {}
    camera_readability: dict[str, dict] = {}
    timestamp_readability: dict[str, dict] = {}
    timestamps_by_stream: dict[str, TimestampSeries] = {}
    timestamp_duplicate_summaries: dict[str, dict] = {}

    for stream in camera_streams:
        timestamp_path = session_dir / "timestamps" / f"{stream}_timestamps.txt"
        try:
            video_path = find_named_video(session_dir / "videos", stream, preferred_ext=video_ext)
        except (FileNotFoundError, FileExistsError):
            video_path = session_dir / "videos" / f"{stream}{video_ext}"
        videos[stream] = str(video_path)
        timestamps[stream] = str(timestamp_path)
        timestamp_paths[stream] = timestamp_path
        modality_presence[f"videos:{stream}"] = _presence(video_path, required=True)
        modality_presence[f"timestamps:{stream}"] = _presence(timestamp_path, required=True)

        try:
            if not video_path.exists():
                raise SessionIntegrityError(f"Missing video: {video_path}")
            info = dict(video_inspector(video_path))
            info.setdefault("path", str(video_path))
            info.setdefault("frame_readable", True)
            info["readable"] = bool(info.get("frame_readable", True))
            camera_readability[stream] = {
                "path": str(video_path),
                "exists": True,
                "readable": bool(info["readable"]),
                "frame_readable": bool(info.get("frame_readable", True)),
                "error": None if info["readable"] else "first frame is not readable",
            }
            for key in ("width", "height", "frame_count", "fps"):
                if key in info:
                    camera_readability[stream][key] = info[key]
            video_infos[stream] = info
            if not info["readable"]:
                collection_errors.append(f"Video first frame is not readable: {video_path}")
        except Exception as exc:
            message = str(exc)
            camera_readability[stream] = {
                "path": str(video_path),
                "exists": video_path.exists(),
                "readable": False,
                "frame_readable": False,
                "error": message,
            }
            video_infos[stream] = {
                "path": str(video_path),
                "exists": video_path.exists(),
                "readable": False,
                "error": message,
            }
            collection_errors.append(message)

        try:
            series = parse_timestamp_file(timestamp_path)
            timestamps_by_stream[stream] = series
            timestamp_duplicate_summaries[stream] = summarize_timestamp_duplicates(series)
            timestamp_readability[stream] = {
                "path": str(timestamp_path),
                "exists": True,
                "readable": True,
                "frame_count": int(len(series.timestamps)),
                "error": None,
            }
        except Exception as exc:
            message = str(exc)
            timestamp_readability[stream] = {
                "path": str(timestamp_path),
                "exists": timestamp_path.exists(),
                "readable": False,
                "frame_count": 0,
                "error": message,
            }
            timestamp_duplicate_summaries[stream] = {
                "frame_count": 0,
                "duplicate_count": 0,
                "duplicate_rate": 0.0,
                "first_duplicate_index": None,
                "error": message,
            }
            collection_errors.append(message)

    readable_video_infos = [
        info
        for info in video_infos.values()
        if info.get("readable", True) and int(info.get("width") or 0) > 0 and int(info.get("height") or 0) > 0
    ]
    width = int(readable_video_infos[0]["width"]) if readable_video_infos else 0
    height = int(readable_video_infos[0]["height"]) if readable_video_infos else 0
    available_counts = [len(series.timestamps) for series in timestamps_by_stream.values()]
    available_counts.extend(
        int(info["frame_count"])
        for info in video_infos.values()
        if info.get("readable", True) and int(info.get("frame_count") or 0) > 0
    )
    available_frames = min(available_counts) if available_counts else 0
    frame_window_num_frames = int(num_frames) if num_frames is not None else max(available_frames - int(start), 0)
    frame_count_consistency = _build_frame_count_consistency(
        camera_streams=camera_streams,
        video_infos=video_infos,
        timestamps_by_stream=timestamps_by_stream,
        available_frames=available_frames,
    )
    repaired_timestamps_by_stream, timestamp_repair_summary = repair_duplicate_timestamps_by_stream(
        timestamps_by_stream,
        float(fps),
    )
    timestamp_alignment_summary: dict[str, dict] = {}
    timestamp_alignment_repaired_summary: dict[str, dict] = {}
    if MASTER_STREAM not in timestamps_by_stream:
        collection_errors.append(f"Master stream {MASTER_STREAM!r} timestamp is not readable")
    elif frame_window_num_frames <= 0:
        collection_errors.append(f"num_frames must be positive, got {frame_window_num_frames}")
    elif int(start) + frame_window_num_frames > available_frames:
        collection_errors.append(
            f"Requested frames [{int(start)}, {int(start) + frame_window_num_frames}) exceed available frame count {available_frames}"
        )
    elif set(camera_streams).issubset(timestamps_by_stream):
        try:
            timestamp_alignment_summary = assert_aligned_timestamps(
                timestamps_by_stream,
                MASTER_STREAM,
                int(start),
                frame_window_num_frames,
                float(timestamp_tolerance_s),
            )
            if timestamp_repair_summary["applied"]:
                timestamp_alignment_repaired_summary = assert_aligned_timestamps(
                    repaired_timestamps_by_stream,
                    MASTER_STREAM,
                    int(start),
                    frame_window_num_frames,
                    float(timestamp_tolerance_s),
                )
        except Exception as exc:
            collection_errors.append(str(exc))
    else:
        collection_errors.append("Cannot compute timestamp alignment because one or more timestamp streams are unreadable")

    timestamp_file_repair = maybe_repair_timestamp_files(
        timestamp_paths=timestamp_paths,
        timestamp_duplicate_summaries=timestamp_duplicate_summaries,
        repaired_timestamps_by_stream=repaired_timestamps_by_stream,
        timestamp_alignment_repaired_summary=timestamp_alignment_repaired_summary,
        repair_timestamp_files=repair_timestamp_files,
        max_repair_duplicate_rate=float(max_repair_duplicate_rate),
        max_repair_consecutive_duplicate_frames=int(max_repair_consecutive_duplicate_frames),
    )

    try:
        imu_summary = inspect_imu(imu_path)
    except Exception as exc:
        imu_summary = {"path": str(imu_path), "readable": False, "error": str(exc)}
        collection_errors.append(str(exc))

    if isinstance(imu_summary, dict) and imu_summary.get("first_timestamp") is not None and timestamps_by_stream:
        imu_start = float(imu_summary["first_timestamp"])
        imu_end = float(imu_summary["last_timestamp"])
        last_frame_idx = int(start) + frame_window_num_frames - 1
        if frame_window_num_frames > 0:
            imu_summary["start_delta_to_cameras_s"] = {
                stream: float(imu_start - series.timestamps[int(start)])
                for stream, series in timestamps_by_stream.items()
                if int(start) < len(series.timestamps)
            }
            imu_summary["end_delta_to_cameras_s"] = {
                stream: float(imu_end - series.timestamps[last_frame_idx])
                for stream, series in timestamps_by_stream.items()
                if last_frame_idx < len(series.timestamps)
            }
    imu_time_coverage = _build_imu_time_coverage(
        imu_summary=imu_summary,
        timestamps_by_stream=timestamps_by_stream,
        start=int(start),
        num_frames=frame_window_num_frames,
        fps=fps,
    )

    try:
        audio_summary = inspect_audio(audio_path)
        if audio_summary is None:
            collection_errors.append(f"Missing audio: {audio_path}")
    except Exception as exc:
        audio_summary = {
            "path": str(audio_path),
            "exists": audio_path.exists(),
            "readable": False,
            "error": str(exc),
        }
        collection_errors.append(str(exc))
    audio_video_duration_consistency = _build_audio_video_duration_consistency(
        audio_summary=audio_summary,
        video_infos=video_infos,
    )

    return {
        "mode": "quality_report",
        "collection_status": "completed_with_errors" if collection_errors else "complete",
        "collection_errors": collection_errors,
        "source_session": str(session_dir),
        "environment": environment_name,
        "report_time_utc": datetime.now(timezone.utc).isoformat(),
        "camera_streams": list(camera_streams),
        "frame_window": {
            "start": int(start),
            "num_frames": int(frame_window_num_frames),
            "available_frames": int(available_frames),
        },
        "fps": int(fps),
        "meta": meta,
        "image_size": {
            "width": int(width),
            "height": int(height),
        },
        "modality_presence": modality_presence,
        "camera_readability": camera_readability,
        "timestamp_readability": timestamp_readability,
        "frame_count_consistency": frame_count_consistency,
        "imu_time_coverage": imu_time_coverage,
        "audio_video_duration_consistency": audio_video_duration_consistency,
        "videos": videos,
        "timestamps": timestamps,
        "video_infos": video_infos,
        "timestamp_duplicates": timestamp_duplicate_summaries,
        "timestamp_alignment": timestamp_alignment_summary,
        "timestamp_repair": timestamp_repair_summary,
        "timestamp_alignment_repaired": timestamp_alignment_repaired_summary,
        "timestamp_file_repair": timestamp_file_repair,
        "imu": imu_summary,
        "audio": audio_summary,
        "timestamp_tolerance_s": float(timestamp_tolerance_s),
        "notes": [
            "Raw quality report is resilient: source errors are recorded in this JSON instead of interrupting the pipeline.",
            "Video check opens the container and attempts to decode the first frame.",
            "Alignment statistics are recorded for review and evaluated by raw_headband_quality_gate.",
            "IMU orientation quaternion columns are ignored by signal quality checks.",
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
    repair_timestamp_files: bool = False,
    max_repair_duplicate_rate: float = 0.001,
    max_repair_consecutive_duplicate_frames: int = 5,
    video_inspector=inspect_video,
) -> Path:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    annotation_dir = output_root / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    report_path = annotation_dir / "quality_report.json"
    if report_path.exists() and not overwrite:
        raise SessionIntegrityError(f"Quality report exists; pass --overwrite to replace it: {report_path}")
    report = build_resilient_quality_report(
        session_dir=session_dir,
        camera_streams=tuple(camera_streams),
        start=start,
        num_frames=num_frames,
        timestamp_tolerance_s=timestamp_tolerance_s,
        video_ext=video_ext,
        environment_name=environment_name,
        repair_timestamp_files=repair_timestamp_files,
        max_repair_duplicate_rate=max_repair_duplicate_rate,
        max_repair_consecutive_duplicate_frames=max_repair_consecutive_duplicate_frames,
        video_inspector=video_inspector,
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
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
    parser.add_argument("--repair_timestamp_files", action="store_true")
    parser.add_argument("--max_repair_duplicate_rate", type=float, default=0.001)
    parser.add_argument("--max_repair_consecutive_duplicate_frames", type=int, default=5)
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
        repair_timestamp_files=args.repair_timestamp_files,
        max_repair_duplicate_rate=args.max_repair_duplicate_rate,
        max_repair_consecutive_duplicate_frames=args.max_repair_consecutive_duplicate_frames,
        overwrite=args.overwrite,
    )
    print(json.dumps({"report_path": str(report_path), "mode": "quality_report"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
