"""Raw session validation and timestamp synchronization for VIO."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

IMU_FIELDS = (
    "timestamp",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
)


@dataclass(frozen=True)
class CameraTimestamp:
    frame_index: int
    timestamp: float


@dataclass(frozen=True)
class StereoPair:
    timestamp: float
    left_frame_index: int
    right_frame_index: int
    left_timestamp: float
    right_timestamp: float

    @property
    def delta_s(self) -> float:
        return self.left_timestamp - self.right_timestamp


@dataclass(frozen=True)
class StereoMatchResult:
    pairs: tuple[StereoPair, ...]
    left_source_count: int
    right_source_count: int
    left_duplicate_count: int
    right_duplicate_count: int

    @property
    def left_unmatched_count(self) -> int:
        return self.left_source_count - len(self.pairs)

    @property
    def right_unmatched_count(self) -> int:
        return self.right_source_count - len(self.pairs)


@dataclass(frozen=True)
class ImuSample:
    timestamp: float
    angular_velocity: tuple[float, float, float]
    linear_acceleration: tuple[float, float, float]


@dataclass(frozen=True)
class PreparedSession:
    session_dir: Path
    output_dir: Path
    stereo_pairs_path: Path
    imu_clean_path: Path
    report_path: Path


def load_camera_timestamps(path: str | Path) -> list[CameraTimestamp]:
    """Load either `<index> <timestamp>` or one-timestamp-per-line files."""
    source = Path(path)
    samples: list[CameraTimestamp] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        columns = line.replace(",", " ").split()
        if len(columns) == 1:
            frame_index = len(samples)
            timestamp_text = columns[0]
        elif len(columns) >= 2:
            frame_index = int(columns[0])
            timestamp_text = columns[1]
        else:
            raise ValueError(f"Invalid timestamp row at {source}:{line_number}")
        timestamp = float(timestamp_text)
        if not math.isfinite(timestamp):
            raise ValueError(f"Non-finite timestamp at {source}:{line_number}")
        if samples and timestamp < samples[-1].timestamp:
            raise ValueError(
                f"Camera timestamps go backwards at {source}:{line_number}"
            )
        samples.append(CameraTimestamp(frame_index=frame_index, timestamp=timestamp))
    if not samples:
        raise ValueError(f"No camera timestamps found in {source}")
    return samples


def _deduplicate_timestamps(
    samples: Iterable[CameraTimestamp],
) -> tuple[list[CameraTimestamp], int]:
    unique: list[CameraTimestamp] = []
    seen: set[float] = set()
    duplicate_count = 0
    for sample in samples:
        if sample.timestamp in seen:
            duplicate_count += 1
            continue
        seen.add(sample.timestamp)
        unique.append(sample)
    return unique, duplicate_count


def match_stereo_timestamps(
    left: list[CameraTimestamp],
    right: list[CameraTimestamp],
    *,
    tolerance_s: float = 0.001,
) -> StereoMatchResult:
    """Greedily form ordered one-to-one stereo pairs by capture time."""
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    left_unique, left_duplicates = _deduplicate_timestamps(left)
    right_unique, right_duplicates = _deduplicate_timestamps(right)
    pairs: list[StereoPair] = []
    left_index = 0
    right_index = 0
    while left_index < len(left_unique) and right_index < len(right_unique):
        left_sample = left_unique[left_index]
        right_sample = right_unique[right_index]
        delta_s = left_sample.timestamp - right_sample.timestamp
        if abs(delta_s) <= tolerance_s:
            pairs.append(
                StereoPair(
                    timestamp=left_sample.timestamp,
                    left_frame_index=left_sample.frame_index,
                    right_frame_index=right_sample.frame_index,
                    left_timestamp=left_sample.timestamp,
                    right_timestamp=right_sample.timestamp,
                )
            )
            left_index += 1
            right_index += 1
        elif delta_s < 0:
            left_index += 1
        else:
            right_index += 1
    if not pairs:
        raise ValueError(f"No stereo timestamps matched within {tolerance_s:.6f}s")
    return StereoMatchResult(
        pairs=tuple(pairs),
        left_source_count=len(left),
        right_source_count=len(right),
        left_duplicate_count=left_duplicates,
        right_duplicate_count=right_duplicates,
    )


def load_imu_samples(path: str | Path) -> tuple[list[ImuSample], dict[str, Any]]:
    """Load the OpenVINS IMU fields and discard repeated timestamps."""
    source = Path(path)
    samples: list[ImuSample] = []
    seen: set[float] = set()
    duplicate_count = 0
    source_count = 0
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            field for field in IMU_FIELDS if field not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"IMU CSV missing fields: {', '.join(missing)}")
        previous_timestamp: float | None = None
        for row_number, row in enumerate(reader, start=2):
            source_count += 1
            timestamp = float(row["timestamp"])
            if not math.isfinite(timestamp):
                raise ValueError(f"Non-finite IMU timestamp at {source}:{row_number}")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    f"IMU timestamps go backwards at {source}:{row_number}"
                )
            previous_timestamp = timestamp
            if timestamp in seen:
                duplicate_count += 1
                continue
            seen.add(timestamp)
            values = [float(row[field]) for field in IMU_FIELDS[1:]]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite IMU value at {source}:{row_number}")
            samples.append(
                ImuSample(
                    timestamp=timestamp,
                    angular_velocity=tuple(values[:3]),
                    linear_acceleration=tuple(values[3:]),
                )
            )
    if not samples:
        raise ValueError(f"No IMU samples found in {source}")
    deltas = [right.timestamp - left.timestamp for left, right in pairwise(samples)]
    positive_deltas = [delta for delta in deltas if delta > 0]
    report = {
        "source_path": str(source.resolve()),
        "sample_count_source": source_count,
        "sample_count_clean": len(samples),
        "duplicate_count": duplicate_count,
        "first_timestamp": samples[0].timestamp,
        "last_timestamp": samples[-1].timestamp,
        "duration_s": samples[-1].timestamp - samples[0].timestamp,
        "median_period_s": sorted(positive_deltas)[len(positive_deltas) // 2]
        if positive_deltas
        else None,
        "max_period_s": max(positive_deltas) if positive_deltas else None,
    }
    return samples, report


def _write_stereo_pairs(path: Path, pairs: Iterable[StereoPair]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "left_frame_index",
                "right_frame_index",
                "left_timestamp",
                "right_timestamp",
                "delta_s",
            ]
        )
        for pair in pairs:
            writer.writerow(
                [
                    f"{pair.timestamp:.9f}",
                    pair.left_frame_index,
                    pair.right_frame_index,
                    f"{pair.left_timestamp:.9f}",
                    f"{pair.right_timestamp:.9f}",
                    f"{pair.delta_s:.9f}",
                ]
            )


def _write_imu(path: Path, samples: Iterable[ImuSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(IMU_FIELDS)
        for sample in samples:
            writer.writerow(
                [
                    f"{sample.timestamp:.9f}",
                    *sample.angular_velocity,
                    *sample.linear_acceleration,
                ]
            )


def prepare_session(
    session_dir: str | Path,
    output_dir: str | Path,
    *,
    tolerance_s: float = 0.001,
) -> PreparedSession:
    """Create deterministic OpenVINS input indexes without copying videos."""
    session = Path(session_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    left_path = session / "timestamps" / "left_timestamps.txt"
    right_path = session / "timestamps" / "right_timestamps.txt"
    imu_path = session / "imu" / "imu0.csv"
    left = load_camera_timestamps(left_path)
    right = load_camera_timestamps(right_path)
    stereo = match_stereo_timestamps(left, right, tolerance_s=tolerance_s)
    imu_samples, imu_report = load_imu_samples(imu_path)

    stereo_pairs_path = output / "stereo_pairs.csv"
    imu_clean_path = output / "imu_clean.csv"
    report_path = output / "input_report.json"
    _write_stereo_pairs(stereo_pairs_path, stereo.pairs)
    _write_imu(imu_clean_path, imu_samples)
    max_abs_delta = max(abs(pair.delta_s) for pair in stereo.pairs)
    report = {
        "schema_version": "1.0",
        "session_dir": str(session),
        "stereo": {
            "left_timestamp_path": str(left_path),
            "right_timestamp_path": str(right_path),
            "tolerance_s": tolerance_s,
            "matched_pair_count": len(stereo.pairs),
            "left_source_count": stereo.left_source_count,
            "right_source_count": stereo.right_source_count,
            "left_duplicate_count": stereo.left_duplicate_count,
            "right_duplicate_count": stereo.right_duplicate_count,
            "left_unmatched_count": stereo.left_unmatched_count,
            "right_unmatched_count": stereo.right_unmatched_count,
            "max_abs_pair_delta_s": max_abs_delta,
            "first_timestamp": stereo.pairs[0].timestamp,
            "last_timestamp": stereo.pairs[-1].timestamp,
        },
        "imu": imu_report,
        "artifacts": {
            "stereo_pairs": str(stereo_pairs_path),
            "imu_clean": str(imu_clean_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PreparedSession(
        session_dir=session,
        output_dir=output,
        stereo_pairs_path=stereo_pairs_path,
        imu_clean_path=imu_clean_path,
        report_path=report_path,
    )
