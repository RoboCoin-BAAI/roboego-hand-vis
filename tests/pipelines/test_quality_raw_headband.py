import json
import logging
from pathlib import Path

import pytest

from roboego_hand_vis.pipelines.quality.nodes import (
    run_raw_headband_integrity_check,
    run_raw_headband_quality_gate,
)


def test_raw_headband_integrity_check_skips_when_disabled():
    result = run_raw_headband_integrity_check({"enabled": False})

    assert result == {
        "stage": "raw_headband_integrity_check",
        "status": "skipped",
        "reason": "disabled",
    }


def test_raw_headband_integrity_check_requires_session_when_enabled():
    with pytest.raises(ValueError, match="session_dir"):
        run_raw_headband_integrity_check({"enabled": True})


def test_raw_headband_integrity_check_calls_exporter(monkeypatch, tmp_path):
    calls = []
    report_path = tmp_path / "session" / "annotation" / "quality_report.json"

    def fake_export_quality_report(**kwargs):
        calls.append(kwargs)
        return report_path

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.quality.nodes.export_quality_report",
        fake_export_quality_report,
    )

    result = run_raw_headband_integrity_check(
        {
            "enabled": True,
            "session_dir": str(tmp_path / "session"),
            "output_root": str(tmp_path / "quality"),
            "camera_streams": ["left", "right"],
            "start": 5,
            "num_frames": 20,
            "timestamp_tolerance_s": 0.02,
            "video_ext": ".mkv",
            "overwrite": True,
            "environment_name": "ego",
        }
    )

    assert calls == [
        {
            "session_dir": tmp_path / "session",
            "output_root": tmp_path / "quality",
            "camera_streams": ("left", "right"),
            "start": 5,
            "num_frames": 20,
            "timestamp_tolerance_s": 0.02,
            "video_ext": ".mkv",
            "overwrite": True,
            "environment_name": "ego",
        }
    ]
    assert result == {
        "stage": "raw_headband_integrity_check",
        "status": "complete",
        "report_path": str(report_path),
    }


def test_raw_headband_integrity_check_accepts_pipe_encoded_camera_streams(monkeypatch, tmp_path):
    calls = []
    report_path = tmp_path / "session" / "annotation" / "quality_report.json"

    def fake_export_quality_report(**kwargs):
        calls.append(kwargs)
        return report_path

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.quality.nodes.export_quality_report",
        fake_export_quality_report,
    )

    run_raw_headband_integrity_check(
        {
            "enabled": True,
            "session_dir": str(tmp_path / "session"),
            "camera_streams": "left|right",
        }
    )

    assert calls[0]["camera_streams"] == ("left", "right")


def test_raw_headband_quality_gate_skips_when_integrity_check_skipped():
    result = run_raw_headband_quality_gate(
        {
            "stage": "raw_headband_integrity_check",
            "status": "skipped",
            "reason": "disabled",
        },
        {"strict": True},
    )

    assert result == {
        "stage": "raw_headband_quality_gate",
        "status": "skipped",
        "reason": "raw_headband_integrity_check_skipped",
    }


def test_raw_headband_quality_gate_records_failed_result_without_raising(tmp_path, capsys):
    report_path = tmp_path / "quality_report.json"
    report_path.write_text(
        """
{
  "timestamp_alignment": {
    "left": {"over_tolerance_count": 0},
    "right": {"over_tolerance_count": 1}
  },
  "timestamp_duplicates": {},
  "imu": {"duplicate_count": 0},
  "video_infos": {},
  "fps": 30
}
""",
        encoding="utf-8",
    )

    result = run_raw_headband_quality_gate(
        {
            "stage": "raw_headband_integrity_check",
            "status": "complete",
            "report_path": str(report_path),
        },
        {"strict": True},
    )

    assert result["status"] == "failed"
    assert result["passed"] is False
    assert result["failed_checks"] == ["timestamp_alignment:right"]
    stdout = capsys.readouterr().out
    assert "RAW QUALITY GATE FAILED" in stdout
    assert "timestamp_alignment:right" in stdout
    assert str(report_path) in stdout
    updated_report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = updated_report["raw_headband_quality_gate"]
    assert gate["stage"] == "raw_headband_quality_gate"
    assert gate["status"] == "failed"
    assert gate["passed"] is False
    assert gate["failed_checks"] == ["timestamp_alignment:right"]
    assert gate["report_path"] == str(report_path)


def test_raw_headband_quality_gate_logs_summary_for_memory_dataset(tmp_path, caplog, capsys):
    report_path = tmp_path / "quality_report.json"
    report_path.write_text(
        """
{
  "timestamp_alignment": {
    "left": {"over_tolerance_count": 0},
    "right": {"over_tolerance_count": 0}
  },
  "timestamp_duplicates": {},
  "imu": {"duplicate_count": 0},
  "video_infos": {},
  "fps": 30
}
""",
        encoding="utf-8",
    )

    caplog.set_level(logging.INFO)
    result = run_raw_headband_quality_gate(
        {
            "stage": "raw_headband_integrity_check",
            "status": "complete",
            "report_path": str(report_path),
        },
        {"strict": True},
    )

    assert result["stage"] == "raw_headband_quality_gate"
    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["report_path"] == str(report_path)
    assert "Raw headband quality gate: status=passed passed=True failed=0 warnings=0" in caplog.text
    assert "Raw headband quality gate: status=passed passed=True failed=0 warnings=0" in capsys.readouterr().out

    updated_report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = updated_report["raw_headband_quality_gate"]
    assert gate["stage"] == "raw_headband_quality_gate"
    assert gate["status"] == "passed"
    assert gate["passed"] is True
    assert gate["failed_checks"] == []
    assert gate["warnings"] == []
    assert gate["report_path"] == str(report_path)
