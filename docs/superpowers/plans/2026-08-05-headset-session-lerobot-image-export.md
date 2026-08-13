# Headband Session LeRobot Image Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested converter that turns a raw headband session into a local LeRobot image dataset under `lerobot_data/`, with source annotations, absolute timestamps, integrity checks, and strict timestamp alignment.

**Architecture:** Add one focused CLI module, `scripts/export/Headband_data_tolerobot21_v1.py`, that exposes small pure helper functions for timestamp parsing, IMU parsing, feature construction, annotation copying, integrity checks, and export orchestration. Unit tests exercise those helpers without depending on the full real dataset; a final smoke command runs the converter in `py312` for a small frame count.

**Tech Stack:** Python 3.12 via `conda run -n py312`, OpenCV, NumPy, LeRobot `LeRobotDataset`, pytest, stdlib CSV/JSON/shutil/pathlib.

---

## File Structure

- Create `scripts/export/Headband_data_tolerobot21_v1.py`
  - CLI entrypoint and importable helper functions.
  - Keeps LeRobot import inside `main()` / `export_session()` so unit tests can run without importing LeRobot.
  - Reads video frames with OpenCV as RGB arrays.
  - Creates LeRobot dataset with `use_videos=False`.
  - Writes `observation.timestamp.absolute` as a LeRobot scalar feature so the absolute source timestamp is stored in the episode parquet, not only in annotation files.
- Create `tests/export/test_headband_data_tolerobot21_v1.py`
  - Tests timestamp parsing, IMU parsing, nearest IMU lookup, timestamp alignment, feature construction, output path resolution, annotation copy, and integrity failures.
- Modify no existing production files unless tests reveal a packaging issue.

## Task 1: Timestamp, IMU, Feature, and Path Helpers

**Files:**
- Create: `tests/export/test_headband_data_tolerobot21_v1.py`
- Create: `scripts/export/Headband_data_tolerobot21_v1.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/export/test_headband_data_tolerobot21_v1.py` with:

```python
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export.Headband_data_tolerobot21_v1 import (
    DEFAULT_CAMERA_STREAMS,
    IMU_NAMES,
    SessionIntegrityError,
    assert_aligned_timestamps,
    build_features,
    default_output_root,
    nearest_imu_vector,
    parse_imu_csv,
    parse_timestamp_file,
)


def write_timestamp_file(path: Path, rows: list[tuple[int, float]]) -> None:
    path.write_text("\n".join(f"{idx} {ts:.9f}" for idx, ts in rows) + "\n", encoding="utf-8")


def write_imu_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["timestamp", *IMU_NAMES]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_parse_timestamp_file_returns_indices_and_absolute_timestamps(tmp_path):
    path = tmp_path / "left_timestamps.txt"
    write_timestamp_file(path, [(0, 1785854682.133340120), (1, 1785854682.166676283)])

    parsed = parse_timestamp_file(path)

    assert parsed.indices.tolist() == [0, 1]
    assert parsed.timestamps.dtype == np.float64
    assert parsed.timestamps.tolist() == pytest.approx([1785854682.133340120, 1785854682.166676283])


def test_parse_timestamp_file_rejects_non_monotonic_timestamps(tmp_path):
    path = tmp_path / "left_timestamps.txt"
    write_timestamp_file(path, [(0, 10.0), (1, 9.0)])

    with pytest.raises(SessionIntegrityError, match="non-monotonic"):
        parse_timestamp_file(path)


def test_parse_imu_csv_and_nearest_imu_vector(tmp_path):
    path = tmp_path / "imu0.csv"
    row0 = {
        "timestamp": 1.00,
        "orientation_x": 0.1,
        "orientation_y": 0.2,
        "orientation_z": 0.3,
        "orientation_w": 0.4,
        "angular_velocity_x": 1.0,
        "angular_velocity_y": 1.1,
        "angular_velocity_z": 1.2,
        "linear_acceleration_x": 9.0,
        "linear_acceleration_y": 9.1,
        "linear_acceleration_z": 9.2,
    }
    row1 = {**row0, "timestamp": 1.10, "orientation_x": 2.0}
    write_imu_csv(path, [row0, row1])

    imu = parse_imu_csv(path)
    vector = nearest_imu_vector(imu, 1.09)

    assert imu.timestamps.tolist() == pytest.approx([1.0, 1.1])
    assert vector.dtype == np.float32
    assert vector.shape == (10,)
    assert vector[0] == pytest.approx(2.0)


def test_assert_aligned_timestamps_rejects_stream_outside_tolerance():
    master = {"left": parse_timestamp_file_from_rows([(0, 10.0), (1, 10.033333)])}
    streams = {
        **master,
        "right": parse_timestamp_file_from_rows([(0, 10.0), (1, 10.200000)]),
    }

    with pytest.raises(SessionIntegrityError, match="right.*frame 1"):
        assert_aligned_timestamps(streams, "left", start=0, num_frames=2, tolerance_s=0.04)


def parse_timestamp_file_from_rows(rows: list[tuple[int, float]]):
    path = Path("/tmp/in-memory-timestamps.txt")
    write_timestamp_file(path, rows)
    return parse_timestamp_file(path)


def test_build_features_uses_image_dtype_absolute_timestamp_imu_and_zero_action():
    features = build_features(DEFAULT_CAMERA_STREAMS, width=640, height=480)

    assert features["observation.images.left"]["dtype"] == "image"
    assert features["observation.images.left"]["shape"] == (480, 640, 3)
    assert features["observation.imu"]["shape"] == (10,)
    assert features["observation.timestamp.absolute"]["dtype"] == "float64"
    assert features["action"]["shape"] == (1,)


def test_default_output_root_uses_lerobot_data_session_name(tmp_path):
    session = tmp_path / "data" / "session_20260804_224442"

    output = default_output_root(session, tmp_path)

    assert output == tmp_path / "lerobot_data" / "session_20260804_224442"
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```bash
pytest tests/export/test_headband_data_tolerobot21_v1.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.export.Headband_data_tolerobot21_v1'`.

- [ ] **Step 3: Implement minimal helper module**

Create `scripts/export/Headband_data_tolerobot21_v1.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
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
IMU_NAMES = (
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "orientation_w",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
)


class SessionIntegrityError(RuntimeError):
    """Raised when a source session cannot be converted safely."""


@dataclass(frozen=True)
class TimestampSeries:
    indices: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class ImuSeries:
    timestamps: np.ndarray
    values: np.ndarray


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve()


def default_output_root(session_dir: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    session_path = Path(session_dir)
    return project_root / "lerobot_data" / session_path.name


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
    if np.any(np.diff(ts_arr) <= 0):
        raise SessionIntegrityError(f"Timestamp file is non-monotonic: {path}")
    return TimestampSeries(indices=index_arr, timestamps=ts_arr)


def parse_imu_csv(path: str | Path) -> ImuSeries:
    path = Path(path)
    if not path.exists():
        raise SessionIntegrityError(f"Missing IMU CSV: {path}")
    timestamps: list[float] = []
    values: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", *IMU_NAMES}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SessionIntegrityError(f"IMU CSV missing columns {sorted(missing)}: {path}")
        for line_no, row in enumerate(reader, start=2):
            try:
                timestamps.append(float(row["timestamp"]))
                values.append([float(row[name]) for name in IMU_NAMES])
            except (TypeError, ValueError) as exc:
                raise SessionIntegrityError(f"Malformed IMU row {line_no} in {path}") from exc
    if not timestamps:
        raise SessionIntegrityError(f"IMU CSV has no usable rows: {path}")
    ts_arr = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(ts_arr) < 0):
        raise SessionIntegrityError(f"IMU timestamps are non-monotonic: {path}")
    return ImuSeries(timestamps=ts_arr, values=np.asarray(values, dtype=np.float32))


def nearest_imu_vector(imu: ImuSeries, timestamp: float) -> np.ndarray:
    pos = int(np.searchsorted(imu.timestamps, timestamp))
    if pos <= 0:
        idx = 0
    elif pos >= len(imu.timestamps):
        idx = len(imu.timestamps) - 1
    else:
        before = pos - 1
        after = pos
        idx = before if abs(timestamp - imu.timestamps[before]) <= abs(imu.timestamps[after] - timestamp) else after
    return imu.values[idx].astype(np.float32, copy=True)


def assert_aligned_timestamps(
    timestamps_by_stream: dict[str, TimestampSeries],
    master_stream: str,
    start: int,
    num_frames: int,
    tolerance_s: float,
) -> None:
    master = timestamps_by_stream[master_stream]
    end = start + num_frames
    if end > len(master.timestamps):
        raise SessionIntegrityError(f"Master timestamp stream {master_stream} has only {len(master.timestamps)} frames")
    master_slice = master.timestamps[start:end]
    for stream, series in timestamps_by_stream.items():
        if end > len(series.timestamps):
            raise SessionIntegrityError(f"Timestamp stream {stream} has only {len(series.timestamps)} frames")
        deltas = np.abs(series.timestamps[start:end] - master_slice)
        bad = np.where(deltas > tolerance_s)[0]
        if len(bad):
            rel = int(bad[0])
            frame_idx = start + rel
            raise SessionIntegrityError(
                f"Timestamp stream {stream} is not aligned at frame {frame_idx}: "
                f"delta={deltas[rel]:.9f}s tolerance={tolerance_s:.9f}s"
            )


def build_features(camera_streams: tuple[str, ...], width: int, height: int) -> dict:
    features = {
        "observation.imu": {
            "dtype": "float32",
            "shape": (10,),
            "names": list(IMU_NAMES),
        },
        "observation.timestamp.absolute": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["timestamp"],
        },
        "action": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["placeholder"],
        },
    }
    for stream in camera_streams:
        features[f"observation.images.{stream}"] = {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    return features
```

- [ ] **Step 4: Run helper tests to verify they pass**

Run:

```bash
pytest tests/export/test_headband_data_tolerobot21_v1.py -v
```

Expected: PASS for helper tests in Task 1.

- [ ] **Step 5: Commit helpers**

Run:

```bash
git add scripts/export/Headband_data_tolerobot21_v1.py tests/export/test_headband_data_tolerobot21_v1.py
git commit -m "feat: add headset export parsing helpers"
```

## Task 2: Source Integrity and Annotation Copy

**Files:**
- Modify: `tests/export/test_headband_data_tolerobot21_v1.py`
- Modify: `scripts/export/Headband_data_tolerobot21_v1.py`

- [ ] **Step 1: Add failing tests for integrity checks and annotations**

Append these imports to the existing import list:

```python
from scripts.export.Headband_data_tolerobot21_v1 import (
    copy_source_annotations,
    inspect_video,
    validate_source_session,
)
```

Append these tests:

```python
def make_minimal_session(tmp_path: Path) -> Path:
    session = tmp_path / "session_20260804_224442"
    (session / "videos").mkdir(parents=True)
    (session / "timestamps").mkdir()
    (session / "imu").mkdir()
    (session / "calibrations").mkdir()
    (session / "audio").mkdir()
    (session / "meta.json").write_text(
        json.dumps({"fps": 30, "topics": {"left": {"written_frames": 2}}}),
        encoding="utf-8",
    )
    (session / "encoder.log").write_text("encoder ok\n", encoding="utf-8")
    (session / "calibrations" / "calibrations.md").write_text("# calibration\n", encoding="utf-8")
    (session / "calibrations" / "camera-camchain.yaml").write_text("cam0: {}\n", encoding="utf-8")
    (session / "calibrations" / "camera_imu-camchain-imucam.yaml").write_text("cam0: {}\n", encoding="utf-8")
    (session / "calibrations" / "camera_imu-imu.yaml").write_text("imu0: {}\n", encoding="utf-8")
    (session / "audio" / "audio.wav").write_bytes(b"fake")
    for stream in DEFAULT_CAMERA_STREAMS:
        write_timestamp_file(session / "timestamps" / f"{stream}_timestamps.txt", [(0, 1.0), (1, 1.033333333)])
    row = {
        "timestamp": 1.0,
        "orientation_x": 0.0,
        "orientation_y": 0.0,
        "orientation_z": 0.0,
        "orientation_w": 1.0,
        "angular_velocity_x": 0.0,
        "angular_velocity_y": 0.0,
        "angular_velocity_z": 0.0,
        "linear_acceleration_x": 0.0,
        "linear_acceleration_y": 9.8,
        "linear_acceleration_z": 0.0,
    }
    write_imu_csv(session / "imu" / "imu0.csv", [row, {**row, "timestamp": 1.033333333}])
    return session


def test_validate_source_session_rejects_missing_video_before_export(tmp_path):
    session = make_minimal_session(tmp_path)

    with pytest.raises(SessionIntegrityError, match="Missing video"):
        validate_source_session(session, DEFAULT_CAMERA_STREAMS, start=0, num_frames=2, timestamp_tolerance_s=0.04)


def test_validate_source_session_rejects_misaligned_camera_timestamps(tmp_path):
    session = make_minimal_session(tmp_path)
    write_timestamp_file(session / "timestamps" / "right_timestamps.txt", [(0, 1.0), (1, 1.2)])

    with pytest.raises(SessionIntegrityError, match="right.*frame 1"):
        validate_source_session(
            session,
            DEFAULT_CAMERA_STREAMS,
            start=0,
            num_frames=2,
            timestamp_tolerance_s=0.04,
            video_inspector=lambda path: {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
        )


def test_validate_source_session_accepts_aligned_mock_videos(tmp_path):
    session = make_minimal_session(tmp_path)

    result = validate_source_session(
        session,
        DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=2,
        timestamp_tolerance_s=0.04,
        video_inspector=lambda path: {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
    )

    assert result.width == 640
    assert result.height == 480
    assert result.fps == 30
    assert result.num_frames == 2


def test_copy_source_annotations_preserves_metadata_timestamps_imu_and_calibrations(tmp_path):
    session = make_minimal_session(tmp_path)
    output_root = tmp_path / "lerobot" / session.name

    written = copy_source_annotations(
        session,
        output_root,
        DEFAULT_CAMERA_STREAMS,
        video_infos={
            "left": {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
            "right": {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
            "bleft": {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
            "bright": {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
        },
        export_num_frames=2,
        environment_name="py312",
    )

    annotation = output_root / "annotation"
    assert (annotation / "source_meta.json").exists()
    assert (annotation / "encoder.log").read_text(encoding="utf-8") == "encoder ok\n"
    assert (annotation / "timestamps" / "left_timestamps.txt").exists()
    assert (annotation / "imu" / "imu0.csv").exists()
    assert (annotation / "calibrations" / "camera-camchain.yaml").exists()
    source_files = json.loads((annotation / "source_files.json").read_text(encoding="utf-8"))
    assert source_files["environment"] == "py312"
    assert source_files["export_num_frames"] == 2
    assert written["source_files"].name == "source_files.json"
```

- [ ] **Step 2: Run new tests to verify missing functions**

Run:

```bash
pytest tests/export/test_headband_data_tolerobot21_v1.py -v
```

Expected: FAIL with import errors for `copy_source_annotations`, `inspect_video`, or `validate_source_session`.

- [ ] **Step 3: Implement integrity and annotation helpers**

Append to `scripts/export/Headband_data_tolerobot21_v1.py`:

```python
@dataclass(frozen=True)
class SourceSession:
    session_dir: Path
    camera_streams: tuple[str, ...]
    timestamps_by_stream: dict[str, TimestampSeries]
    imu: ImuSeries
    video_infos: dict[str, dict]
    width: int
    height: int
    fps: int
    start: int
    num_frames: int
    timestamp_tolerance_s: float


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


def validate_source_session(
    session_dir: str | Path,
    camera_streams: tuple[str, ...],
    start: int,
    num_frames: int | None,
    timestamp_tolerance_s: float | None,
    video_inspector=inspect_video,
) -> SourceSession:
    session_dir = Path(session_dir).resolve()
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
        timestamp_tolerance_s = 1.0 / float(fps)

    timestamps_by_stream = {
        stream: parse_timestamp_file(session_dir / "timestamps" / f"{stream}_timestamps.txt")
        for stream in camera_streams
    }
    if MASTER_STREAM not in timestamps_by_stream:
        raise SessionIntegrityError(f"Master stream {MASTER_STREAM!r} must be enabled")
    imu = parse_imu_csv(session_dir / "imu" / "imu0.csv")
    video_infos = {
        stream: video_inspector(session_dir / "videos" / f"{stream}.mp4")
        for stream in camera_streams
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
            raise SessionIntegrityError(f"Video frame count is lower than export frame count for {stream}")

    assert_aligned_timestamps(timestamps_by_stream, MASTER_STREAM, start, num_frames, float(timestamp_tolerance_s))
    return SourceSession(
        session_dir=session_dir,
        camera_streams=tuple(camera_streams),
        timestamps_by_stream=timestamps_by_stream,
        imu=imu,
        video_infos=video_infos,
        width=width,
        height=height,
        fps=fps,
        start=start,
        num_frames=num_frames,
        timestamp_tolerance_s=float(timestamp_tolerance_s),
    )


def copy_if_exists(source: Path, target: Path) -> Path | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def copy_source_annotations(
    session_dir: str | Path,
    output_root: str | Path,
    camera_streams: tuple[str, ...],
    video_infos: dict[str, dict],
    export_num_frames: int,
    environment_name: str,
) -> dict[str, Path]:
    session_dir = Path(session_dir).resolve()
    output_root = Path(output_root).resolve()
    annotation_dir = output_root / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    copied = copy_if_exists(session_dir / "meta.json", annotation_dir / "source_meta.json")
    if copied:
        written["source_meta"] = copied
    copied = copy_if_exists(session_dir / "encoder.log", annotation_dir / "encoder.log")
    if copied:
        written["encoder_log"] = copied

    for source in (session_dir / "calibrations").glob("*"):
        if source.is_file():
            copy_if_exists(source, annotation_dir / "calibrations" / source.name)
    for stream in camera_streams:
        copy_if_exists(
            session_dir / "timestamps" / f"{stream}_timestamps.txt",
            annotation_dir / "timestamps" / f"{stream}_timestamps.txt",
        )
    copy_if_exists(session_dir / "imu" / "imu0.csv", annotation_dir / "imu" / "imu0.csv")

    source_files = {
        "source_session": str(session_dir),
        "environment": environment_name,
        "export_time_utc": datetime.now(timezone.utc).isoformat(),
        "camera_streams": list(camera_streams),
        "videos": {stream: str(session_dir / "videos" / f"{stream}.mp4") for stream in camera_streams},
        "timestamps": {stream: str(session_dir / "timestamps" / f"{stream}_timestamps.txt") for stream in camera_streams},
        "imu": str(session_dir / "imu" / "imu0.csv"),
        "audio": str(session_dir / "audio" / "audio.wav") if (session_dir / "audio" / "audio.wav").exists() else None,
        "video_infos": video_infos,
        "export_num_frames": int(export_num_frames),
    }
    source_files_path = annotation_dir / "source_files.json"
    source_files_path.write_text(json.dumps(source_files, indent=2, ensure_ascii=False), encoding="utf-8")
    written["source_files"] = source_files_path
    return written
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/export/test_headband_data_tolerobot21_v1.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit integrity and annotation helpers**

Run:

```bash
git add scripts/export/Headband_data_tolerobot21_v1.py tests/export/test_headband_data_tolerobot21_v1.py
git commit -m "feat: validate headset source sessions"
```

## Task 3: LeRobot Export CLI

**Files:**
- Modify: `tests/export/test_headband_data_tolerobot21_v1.py`
- Modify: `scripts/export/Headband_data_tolerobot21_v1.py`

- [ ] **Step 1: Add failing tests for frame payload construction and parser defaults**

Append imports:

```python
from scripts.export.Headband_data_tolerobot21_v1 import (
    build_arg_parser,
    build_frame_payload,
)
```

Append tests:

```python
def test_build_frame_payload_includes_images_imu_absolute_timestamp_and_zero_action():
    imu = ImuSeries(
        timestamps=np.asarray([10.0], dtype=np.float64),
        values=np.asarray([[0, 0, 0, 1, 0.1, 0.2, 0.3, 1, 2, 3]], dtype=np.float32),
    )
    timestamps = {MASTER_STREAM: TimestampSeries(np.asarray([0]), np.asarray([10.0], dtype=np.float64))}
    images = {
        "left": np.zeros((2, 3, 3), dtype=np.uint8),
        "right": np.ones((2, 3, 3), dtype=np.uint8),
    }

    frame = build_frame_payload(("left", "right"), images, timestamps, imu, frame_idx=0)

    assert frame["observation.images.left"].shape == (2, 3, 3)
    assert frame["observation.imu"].tolist() == pytest.approx([0, 0, 0, 1, 0.1, 0.2, 0.3, 1, 2, 3])
    assert frame["observation.timestamp.absolute"].dtype == np.float64
    assert frame["observation.timestamp.absolute"].tolist() == pytest.approx([10.0])
    assert frame["action"].dtype == np.float32
    assert frame["action"].tolist() == [0.0]


def test_arg_parser_defaults_output_root_to_lerobot_data_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = tmp_path / "data" / "session_20260804_224442"
    parser = build_arg_parser()

    args = parser.parse_args(["--session", str(session)])

    assert args.output_root is None
    assert args.camera_stream == list(DEFAULT_CAMERA_STREAMS)
    assert args.repo_id == "local/roboego-headband-session"
```

- [ ] **Step 2: Run tests to verify missing functions**

Run:

```bash
pytest tests/export/test_headband_data_tolerobot21_v1.py -v
```

Expected: FAIL with missing `build_frame_payload` or `build_arg_parser`.

- [ ] **Step 3: Implement frame payload, video reading, export, and CLI**

Append to `scripts/export/Headband_data_tolerobot21_v1.py`:

```python
def read_video_frame(cap: cv2.VideoCapture, stream: str, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise SessionIntegrityError(f"Failed to decode {stream} frame {frame_idx}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_frame_payload(
    camera_streams: tuple[str, ...],
    images_by_stream: dict[str, np.ndarray],
    timestamps_by_stream: dict[str, TimestampSeries],
    imu: ImuSeries,
    frame_idx: int,
) -> dict:
    absolute_ts = float(timestamps_by_stream[MASTER_STREAM].timestamps[frame_idx])
    frame = {
        "observation.imu": nearest_imu_vector(imu, absolute_ts),
        "observation.timestamp.absolute": np.asarray([absolute_ts], dtype=np.float64),
        "action": np.zeros((1,), dtype=np.float32),
    }
    for stream in camera_streams:
        frame[f"observation.images.{stream}"] = images_by_stream[stream]
    return frame


def export_session(
    session_dir: str | Path,
    output_root: str | Path,
    repo_id: str,
    task: str,
    camera_streams: tuple[str, ...] = DEFAULT_CAMERA_STREAMS,
    start: int = 0,
    num_frames: int | None = None,
    timestamp_tolerance_s: float | None = None,
    overwrite: bool = False,
    environment_name: str = "py312",
) -> Path:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("LeRobot is not installed in this Python environment. Run with: conda run -n py312 python ...") from exc

    session = validate_source_session(
        session_dir,
        tuple(camera_streams),
        start=start,
        num_frames=num_frames,
        timestamp_tolerance_s=timestamp_tolerance_s,
    )
    output_root = Path(output_root).resolve()
    if output_root.exists():
        if not overwrite:
            raise SessionIntegrityError(f"Output exists; pass --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    features = build_features(session.camera_streams, width=session.width, height=session.height)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=session.fps,
        features=features,
        root=output_root,
        robot_type="roboego_headband",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    caps = {
        stream: cv2.VideoCapture(str(session.session_dir / "videos" / f"{stream}.mp4"))
        for stream in session.camera_streams
    }
    try:
        for frame_idx in range(session.start, session.start + session.num_frames):
            images = {
                stream: read_video_frame(caps[stream], stream, frame_idx)
                for stream in session.camera_streams
            }
            frame = build_frame_payload(session.camera_streams, images, session.timestamps_by_stream, session.imu, frame_idx)
            relative_ts = float(frame_idx - session.start) / float(session.fps)
            dataset.add_frame(frame, task=task, timestamp=relative_ts)
        dataset.save_episode()
    finally:
        for cap in caps.values():
            cap.release()

    copy_source_annotations(
        session.session_dir,
        output_root,
        session.camera_streams,
        session.video_infos,
        session.num_frames,
        environment_name,
    )
    return output_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a raw RoboEgo headband session to a LeRobot image dataset.")
    parser.add_argument("--session", required=True, help="Raw headband session directory.")
    parser.add_argument("--output_root", default=None, help="Output dataset root. Defaults to lerobot_data/<session_name>.")
    parser.add_argument("--repo_id", default="local/roboego-headband-session")
    parser.add_argument("--task", default="RoboEgo headband synchronized multi-camera and IMU recording.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--timestamp_tolerance_s", type=float, default=None)
    parser.add_argument("--camera_stream", action="append", choices=DEFAULT_CAMERA_STREAMS, default=list(DEFAULT_CAMERA_STREAMS))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cwd = Path.cwd()
    session_dir = resolve_path(args.session, cwd)
    output_root = resolve_path(args.output_root, cwd) if args.output_root else default_output_root(session_dir, PROJECT_ROOT)
    exported = export_session(
        session_dir=session_dir,
        output_root=output_root,
        repo_id=args.repo_id,
        task=args.task,
        camera_streams=tuple(args.camera_stream),
        start=args.start,
        num_frames=args.num_frames,
        timestamp_tolerance_s=args.timestamp_tolerance_s,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output_root": str(exported), "repo_id": args.repo_id}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests**

Run:

```bash
pytest tests/export/test_headband_data_tolerobot21_v1.py -v
```

Expected: PASS.

- [ ] **Step 5: Run py312 import check**

Run:

```bash
conda run -n py312 python -c "from scripts.export.Headband_data_tolerobot21_v1 import build_features, DEFAULT_CAMERA_STREAMS; print(build_features(DEFAULT_CAMERA_STREAMS, 640, 480)['action']['shape'])"
```

Expected output includes:

```text
(1,)
```

- [ ] **Step 6: Commit export CLI**

Run:

```bash
git add scripts/export/Headband_data_tolerobot21_v1.py tests/export/test_headband_data_tolerobot21_v1.py
git commit -m "feat: add headband lerobot image export cli"
```

## Task 4: Smoke Conversion and Reload Verification

**Files:**
- Modify only if smoke run reveals a bug: `scripts/export/Headband_data_tolerobot21_v1.py`
- Modify only if smoke run reveals a missing assertion: `tests/export/test_headband_data_tolerobot21_v1.py`

- [ ] **Step 1: Run full unit tests for export package**

Run:

```bash
pytest tests/export -v
```

Expected: PASS.

- [ ] **Step 2: Run a small real-session smoke export in py312**

Run:

```bash
conda run -n py312 python scripts/export/Headband_data_tolerobot21_v1.py \
  --session data/session_20260804_224442 \
  --output_root lerobot_data/session_20260804_224442_smoke \
  --repo_id local/roboego-headband-session-smoke \
  --num_frames 5 \
  --overwrite
```

Expected: command exits 0 and prints JSON containing:

```json
{
  "output_root": ".../lerobot_data/session_20260804_224442_smoke",
  "repo_id": "local/roboego-headband-session-smoke"
}
```

- [ ] **Step 3: Verify image storage and annotations exist**

Run:

```bash
find lerobot_data/session_20260804_224442_smoke -maxdepth 4 -type f | sort | head -80
```

Expected: output includes LeRobot metadata/data files, image files rather than mp4 video files, and:

```text
annotation/source_meta.json
annotation/source_files.json
annotation/timestamps/left_timestamps.txt
annotation/imu/imu0.csv
annotation/calibrations/camera-camchain.yaml
```

- [ ] **Step 4: Reload the smoke dataset with LeRobot**

Run:

```bash
conda run -n py312 python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/roboego-headband-session-smoke", root="lerobot_data/session_20260804_224442_smoke", episodes=[0], download_videos=False)
print(ds.num_frames)
print(ds.features.keys())
sample = ds[0]
print(sample["observation.timestamp.absolute"])
print(sample["action"])
PY
```

Expected: output includes `5`, the absolute timestamp feature key, and action tensor/value with one zero.

- [ ] **Step 5: If smoke revealed a bug, add a failing test before fixing**

If any smoke command fails due to project code, write a focused failing test in `tests/export/test_headband_data_tolerobot21_v1.py` that reproduces the bug without the full real video. Run the test and verify it fails for the expected reason before modifying production code.

- [ ] **Step 6: Commit smoke-tested fixes or verification note**

If code changed, run:

```bash
git add scripts/export/Headband_data_tolerobot21_v1.py tests/export/test_headband_data_tolerobot21_v1.py
git commit -m "fix: harden headband lerobot smoke export"
```

If no code changed, do not create an empty commit.

## Task 5: Final Verification and User Command

**Files:**
- No required file changes.

- [ ] **Step 1: Run final targeted tests**

Run:

```bash
pytest tests/export -v
```

Expected: PASS.

- [ ] **Step 2: Run final import check in py312**

Run:

```bash
conda run -n py312 python -c "import lerobot; from scripts.export.Headband_data_tolerobot21_v1 import export_session; print(getattr(lerobot, '__version__', 'unknown')); print(export_session.__name__)"
```

Expected output includes:

```text
0.3.3
export_session
```

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: only pre-existing unrelated changes remain, or nothing if the workspace was otherwise clean.

- [ ] **Step 4: Report the production conversion command**

Give the user this command for the full conversion:

```bash
conda run -n py312 python scripts/export/Headband_data_tolerobot21_v1.py \
  --session data/session_20260804_224442 \
  --output_root lerobot_data/session_20260804_224442 \
  --repo_id local/roboego-headband-session \
  --overwrite
```

Mention that the converter fails before writing if required files are missing, videos are too short, timestamps are non-monotonic, or camera timestamps exceed the alignment tolerance.

---

## Self-Review

- Spec coverage: The plan covers image-based LeRobot export, `py312`, `lerobot_data/`, annotation copying, zero placeholder action, IMU nearest sample, absolute timestamp feature, source integrity checks, strict timestamp alignment, CLI options, and smoke verification.
- Placeholder scan: No `TODO`, `TBD`, or vague "add appropriate" steps are present; each code step includes exact code and commands.
- Type consistency: Helper names, dataclasses, feature keys, and CLI option names are consistent across tests and implementation snippets.
