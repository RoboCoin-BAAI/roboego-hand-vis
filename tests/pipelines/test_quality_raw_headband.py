import logging
from pathlib import Path

import pytest

from roboego_hand_vis.quality.raw_headband_integrity import SessionIntegrityError
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


def test_raw_headband_quality_gate_raises_when_strict_gate_fails(tmp_path):
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

    with pytest.raises(SessionIntegrityError, match="timestamp_alignment:right"):
        run_raw_headband_quality_gate(
            {
                "stage": "raw_headband_integrity_check",
                "status": "complete",
                "report_path": str(report_path),
            },
            {"strict": True},
        )


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
    assert "Raw headband quality gate: status=passed passed=True failed=0 warnings=0" in caplog.text
    assert "Raw headband quality gate: status=passed passed=True failed=0 warnings=0" in capsys.readouterr().out
