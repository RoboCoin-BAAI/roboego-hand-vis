import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.raw_headband_integrity import (
    DEFAULT_CAMERA_STREAMS,
    SessionIntegrityError,
    TimestampSeries,
    assert_aligned_timestamps,
    build_arg_parser,
    default_quality_report_root,
    export_quality_report,
    inspect_imu,
    parse_timestamp_file,
    summarize_timestamp_duplicates,
    validate_source_session,
)


def write_timestamp_file(path: Path, rows: list[tuple[int, float]]) -> None:
    path.write_text("\n".join(f"{idx} {ts:.9f}" for idx, ts in rows) + "\n", encoding="utf-8")


def parse_timestamp_file_from_rows(rows: list[tuple[int, float]]):
    path = Path("/tmp/in-memory-headband-timestamps.txt")
    write_timestamp_file(path, rows)
    return parse_timestamp_file(path)


def make_minimal_session(tmp_path: Path) -> Path:
    session = tmp_path / "session_20260804_224442"
    (session / "videos").mkdir(parents=True)
    (session / "timestamps").mkdir()
    (session / "imu").mkdir()
    (session / "meta.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
    (session / "imu" / "imu0.csv").write_text(
        "timestamp,angular_velocity_x\n"
        "0.990000000,1.0\n"
        "0.995000000,1.1\n"
        "0.995000000,1.2\n"
        "1.000000000,1.3\n",
        encoding="utf-8",
    )
    for stream in DEFAULT_CAMERA_STREAMS:
        (session / "videos" / f"{stream}.mp4").write_bytes(b"fake mp4")
        write_timestamp_file(session / "timestamps" / f"{stream}_timestamps.txt", [(0, 1.0), (1, 1.033333333)])
    return session


def make_minimal_session_with_video_ext(tmp_path: Path, video_ext: str) -> Path:
    session = make_minimal_session(tmp_path)
    for stream in DEFAULT_CAMERA_STREAMS:
        (session / "videos" / f"{stream}.mp4").rename(session / "videos" / f"{stream}{video_ext}")
    return session


def test_parse_timestamp_file_returns_indices_and_absolute_timestamps(tmp_path):
    path = tmp_path / "left_timestamps.txt"
    write_timestamp_file(path, [(0, 1785854682.133340120), (1, 1785854682.166676283)])

    parsed = parse_timestamp_file(path)

    assert parsed.indices.tolist() == [0, 1]
    assert parsed.timestamps.dtype == np.float64
    assert parsed.timestamps.tolist() == pytest.approx([1785854682.133340120, 1785854682.166676283])


def test_parse_timestamp_file_allows_duplicate_timestamps_and_reports_duplicate_rate(tmp_path):
    path = tmp_path / "left_timestamps.txt"
    write_timestamp_file(path, [(0, 10.0), (1, 10.0), (2, 10.1), (3, 10.1)])

    summary = summarize_timestamp_duplicates(parse_timestamp_file(path))

    assert summary["frame_count"] == 4
    assert summary["duplicate_count"] == 2
    assert summary["duplicate_rate"] == pytest.approx(2 / 3)
    assert summary["first_duplicate_index"] == 1
    assert summary["max_consecutive_duplicate_frames"] == 2
    assert summary["first_max_consecutive_duplicate_index"] == 0


def test_summarize_timestamp_duplicates_reports_longest_consecutive_duplicate_run(tmp_path):
    path = tmp_path / "left_timestamps.txt"
    write_timestamp_file(
        path,
        [
            (0, 10.0),
            (1, 10.1),
            (2, 10.1),
            (3, 10.1),
            (4, 10.1),
            (5, 10.1),
            (6, 10.1),
            (7, 10.2),
        ],
    )

    summary = summarize_timestamp_duplicates(parse_timestamp_file(path))

    assert summary["duplicate_count"] == 5
    assert summary["max_consecutive_duplicate_frames"] == 6
    assert summary["first_max_consecutive_duplicate_index"] == 1


def test_parse_timestamp_file_rejects_decreasing_timestamps(tmp_path):
    path = tmp_path / "left_timestamps.txt"
    write_timestamp_file(path, [(0, 10.0), (1, 9.0)])

    with pytest.raises(SessionIntegrityError, match="decreases"):
        parse_timestamp_file(path)


def test_assert_aligned_timestamps_reports_stream_outside_tolerance_without_raising():
    streams = {
        "left": parse_timestamp_file_from_rows([(0, 10.0), (1, 10.033333)]),
        "right": parse_timestamp_file_from_rows([(0, 10.0), (1, 10.200000)]),
    }

    summary = assert_aligned_timestamps(streams, "left", start=0, num_frames=2, tolerance_s=0.04)

    assert summary["right"]["frame_count"] == 2
    assert summary["right"]["over_tolerance_count"] == 1
    assert summary["right"]["over_tolerance_rate"] == pytest.approx(0.5)
    assert summary["right"]["first_over_tolerance_index"] == 1
    assert summary["right"]["max_abs_delta_s"] == pytest.approx(0.166667)


def test_default_quality_report_root_uses_source_session_directory(tmp_path):
    session = tmp_path / "data" / "session_20260804_224442"

    assert default_quality_report_root(session) == session.resolve()


def test_inspect_imu_reports_fps_duplicate_rate_and_first_timestamp(tmp_path):
    session = make_minimal_session(tmp_path)

    summary = inspect_imu(session / "imu" / "imu0.csv")

    assert summary["path"] == str((session / "imu" / "imu0.csv").resolve())
    assert summary["sample_count"] == 4
    assert summary["first_timestamp"] == pytest.approx(0.99)
    assert summary["last_timestamp"] == pytest.approx(1.0)
    assert summary["duplicate_count"] == 1
    assert summary["duplicate_rate"] == pytest.approx(1 / 3)
    assert summary["estimated_fps"] == pytest.approx(200.0)
    assert summary["signal_quality"]["numeric_column_count"] == 1
    assert summary["signal_quality"]["all_zero_columns"] == []


def test_inspect_imu_reports_all_zero_frozen_and_jump_columns(tmp_path):
    imu_path = tmp_path / "imu0.csv"
    imu_path.write_text(
        "timestamp,gyro_x,gyro_y,accel_z,temperature,orientation_w,orientation_x,orientation_y,orientation_z\n"
        "0.000,0,1.0,0.0,25.0,0,0,0,0\n"
        "0.005,0,1.0,0.1,25.0,0,0,0,0\n"
        "0.010,0,1.0,0.2,25.0,0,0,0,0\n"
        "0.015,0,1.0,100.0,25.0,0,0,0,0\n",
        encoding="utf-8",
    )

    summary = inspect_imu(imu_path)

    quality = summary["signal_quality"]
    assert quality["numeric_column_count"] == 4
    assert quality["ignored_columns"] == ["orientation_w", "orientation_x", "orientation_y", "orientation_z"]
    assert quality["finite_rate"] == pytest.approx(1.0)
    assert quality["all_zero_columns"] == ["gyro_x"]
    assert quality["frozen_columns"] == ["gyro_x", "gyro_y", "temperature"]
    assert quality["jump_columns"] == ["accel_z"]
    assert quality["columns"]["accel_z"]["max_abs_delta"] == pytest.approx(99.8)
    assert quality["columns"]["accel_z"]["max_jump_sample_index"] == 3


def test_inspect_imu_skips_non_numeric_signal_columns(tmp_path):
    imu_path = tmp_path / "imu0.csv"
    imu_path.write_text(
        "timestamp,gyro_x,status\n"
        "0.000,0.1,ok\n"
        "0.005,0.2,ok\n",
        encoding="utf-8",
    )

    quality = inspect_imu(imu_path)["signal_quality"]

    assert quality["numeric_column_count"] == 1
    assert list(quality["columns"]) == ["gyro_x"]


def test_validate_source_session_rejects_missing_video_in_strict_validator(tmp_path):
    session = make_minimal_session(tmp_path)
    (session / "videos" / "left.mp4").unlink()

    with pytest.raises(SessionIntegrityError, match="Missing video"):
        validate_source_session(session, DEFAULT_CAMERA_STREAMS, start=0, num_frames=2, timestamp_tolerance_s=0.04)


def test_export_quality_report_records_missing_video_without_raising(tmp_path):
    session = make_minimal_session(tmp_path)
    (session / "videos" / "left.mp4").unlink()
    output_root = tmp_path / "quality" / session.name

    report_path = export_quality_report(
        session,
        output_root,
        camera_streams=DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=2,
        timestamp_tolerance_s=0.04,
        overwrite=False,
        environment_name="py312",
        video_inspector=lambda path: {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["modality_presence"]["videos:left"]["exists"] is False
    assert report["camera_readability"]["left"]["readable"] is False
    assert "Missing video" in report["camera_readability"]["left"]["error"]
    assert report["timestamp_readability"]["left"]["readable"] is True
    assert report["collection_status"] == "completed_with_errors"


def test_validate_source_session_uses_configured_video_extension(tmp_path):
    session = make_minimal_session_with_video_ext(tmp_path, ".mkv")

    result = validate_source_session(
        session,
        DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=2,
        timestamp_tolerance_s=0.04,
        video_ext=".mkv",
        video_inspector=lambda path: {
            "path": str(path),
            "width": 640,
            "height": 480,
            "frame_count": 2,
            "fps": 30.0,
        },
    )

    assert result.video_infos["left"]["path"].endswith("videos/left.mkv")


def test_validate_source_session_records_duplicate_rates_and_alignment_without_rejecting(tmp_path):
    session = make_minimal_session(tmp_path)
    write_timestamp_file(session / "timestamps" / "bleft_timestamps.txt", [(0, 1.0), (1, 1.0)])
    write_timestamp_file(session / "timestamps" / "right_timestamps.txt", [(0, 1.0), (1, 1.2)])

    result = validate_source_session(
        session,
        DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=2,
        timestamp_tolerance_s=0.04,
        video_inspector=lambda path: {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
    )

    assert result.timestamp_duplicate_summaries["bleft"]["duplicate_rate"] == pytest.approx(1.0)
    assert result.timestamp_alignment_summary["right"]["over_tolerance_count"] == 1


def test_validate_source_session_default_tolerance_allows_one_frame_plus_rounding(tmp_path):
    session = make_minimal_session(tmp_path)

    result = validate_source_session(
        session,
        DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=2,
        timestamp_tolerance_s=None,
        video_inspector=lambda path: {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
    )

    assert result.timestamp_tolerance_s == pytest.approx((1.0 / 30.0) + 0.001)


def test_arg_parser_defaults_to_quality_report_in_source_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = tmp_path / "data" / "session_20260804_224442"

    args = build_arg_parser().parse_args(["--session", str(session)])

    assert args.output_root is None
    assert args.camera_stream == list(DEFAULT_CAMERA_STREAMS)


def test_export_quality_report_writes_only_annotation_report_without_dataset_layout(tmp_path):
    session = make_minimal_session(tmp_path)
    output_root = tmp_path / "quality" / session.name

    report_path = export_quality_report(
        session,
        output_root,
        camera_streams=DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=2,
        timestamp_tolerance_s=0.04,
        overwrite=False,
        environment_name="py312",
        video_inspector=lambda path: {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
    )

    assert report_path == (output_root / "annotation" / "quality_report.json").resolve()
    assert not (output_root / "meta").exists()
    assert not (output_root / "data").exists()
    assert not (output_root / "videos").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "quality_report"
    assert report["frame_window"]["start"] == 0
    assert report["frame_window"]["num_frames"] == 2
    assert report["frame_window"]["available_frames"] == 2
    assert report["timestamp_duplicates"]["left"]["duplicate_rate"] == 0.0
    assert report["timestamp_alignment"]["right"]["over_tolerance_count"] == 0
    assert report["imu"]["sample_count"] == 4
    assert report["imu"]["duplicate_count"] == 1
    assert report["imu"]["start_delta_to_cameras_s"]["left"] == pytest.approx(-0.01)
    assert report["imu"]["end_delta_to_cameras_s"]["left"] == pytest.approx(-0.033333333)
    assert report["frame_count_consistency"] == {
        "video_frame_counts": {"left": 2, "right": 2, "bleft": 2, "bright": 2},
        "timestamp_frame_counts": {"left": 2, "right": 2, "bleft": 2, "bright": 2},
        "video_timestamp_frame_count_deltas": {"left": 0, "right": 0, "bleft": 0, "bright": 0},
        "max_video_frame_count_delta": 0,
        "max_timestamp_frame_count_delta": 0,
        "max_video_timestamp_count_delta": 0,
        "available_frames": 2,
        "policy": "trim_to_min_available_frames",
    }
    assert report["imu_time_coverage"]["computed"] is True
    assert report["imu_time_coverage"]["max_missing_gap_frames"] == pytest.approx(1.0)
    assert report["imu_time_coverage"]["streams"]["left"]["end_missing_gap_frames"] == pytest.approx(1.0)


def test_export_quality_report_records_repaired_timestamp_alignment_for_duplicate_runs(tmp_path):
    session = make_minimal_session(tmp_path)
    left_rows = [(0, 1.0), (1, 1.033333333), (2, 1.066666667), (3, 1.1)]
    right_rows = [(0, 1.0), (1, 1.033333333), (2, 1.033333333), (3, 1.1)]
    for stream in DEFAULT_CAMERA_STREAMS:
        write_timestamp_file(session / "timestamps" / f"{stream}_timestamps.txt", left_rows)
    write_timestamp_file(session / "timestamps" / "right_timestamps.txt", right_rows)

    report_path = export_quality_report(
        session,
        session,
        camera_streams=DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=4,
        timestamp_tolerance_s=0.005,
        overwrite=True,
        repair_timestamp_files=True,
        max_repair_duplicate_rate=1.0,
        max_repair_consecutive_duplicate_frames=5,
        video_inspector=lambda path: {"width": 640, "height": 480, "frame_count": 4, "fps": 30.0},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    repaired_right = parse_timestamp_file(session / "timestamps" / "right_timestamps.txt")

    assert report["timestamp_alignment"]["right"]["over_tolerance_count"] == 1
    assert report["timestamp_repair"]["applied"] is True
    assert report["timestamp_repair"]["streams"]["right"]["max_consecutive_duplicate_frames"] == 2
    assert report["timestamp_alignment_repaired"]["right"]["over_tolerance_count"] == 0
    assert report["timestamp_file_repair"]["applied"] is True
    assert report["timestamp_file_repair"]["streams"]["right"]["written"] is True
    assert repaired_right.timestamps.tolist() == pytest.approx([row[1] for row in left_rows])


def test_export_quality_report_does_not_rewrite_long_duplicate_runs(tmp_path):
    session = make_minimal_session(tmp_path)
    rows = [
        (0, 1.0),
        (1, 1.033333333),
        (2, 1.066666667),
        (3, 1.1),
        (4, 1.133333333),
        (5, 1.166666667),
        (6, 1.2),
        (7, 1.233333333),
    ]
    bad_rows = [
        (0, 1.0),
        (1, 1.033333333),
        (2, 1.033333333),
        (3, 1.033333333),
        (4, 1.033333333),
        (5, 1.033333333),
        (6, 1.033333333),
        (7, 1.233333333),
    ]
    for stream in DEFAULT_CAMERA_STREAMS:
        write_timestamp_file(session / "timestamps" / f"{stream}_timestamps.txt", rows)
    write_timestamp_file(session / "timestamps" / "right_timestamps.txt", bad_rows)

    report_path = export_quality_report(
        session,
        session,
        camera_streams=DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=8,
        timestamp_tolerance_s=0.005,
        overwrite=True,
        repair_timestamp_files=True,
        max_repair_duplicate_rate=1.0,
        max_repair_consecutive_duplicate_frames=5,
        video_inspector=lambda path: {"width": 640, "height": 480, "frame_count": 8, "fps": 30.0},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    unrepaired_right = parse_timestamp_file(session / "timestamps" / "right_timestamps.txt")

    assert report["timestamp_file_repair"]["applied"] is False
    assert "right" in report["timestamp_file_repair"]["skipped_streams"]
    assert unrepaired_right.timestamps.tolist() == pytest.approx([row[1] for row in bad_rows])


def test_export_quality_report_records_configured_video_extension(tmp_path):
    session = make_minimal_session_with_video_ext(tmp_path, ".mkv")
    output_root = tmp_path / "quality" / session.name

    report_path = export_quality_report(
        session,
        output_root,
        camera_streams=DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=2,
        timestamp_tolerance_s=0.04,
        video_ext=".mkv",
        overwrite=False,
        environment_name="py312",
        video_inspector=lambda path: {"width": 640, "height": 480, "frame_count": 2, "fps": 30.0},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["videos"]["left"].endswith("videos/left.mkv")
