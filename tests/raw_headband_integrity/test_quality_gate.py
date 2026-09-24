from __future__ import annotations

from pathlib import Path

from roboego_hand_vis.raw_headband_integrity import cli
from roboego_hand_vis.raw_headband_integrity.rules_logic import evaluate_quality_report


def test_default_audio_video_duration_allows_raw_capture_tail_slack() -> None:
    report = {
        "source_session": "/tmp/session",
        "imu": {
            "duplicate_rate": 0.0,
            "signal_quality": {
                "all_zero_columns": [],
                "frozen_columns": [],
                "jump_columns": [],
            },
        },
        "audio": {"readable": True},
        "audio_video_duration_consistency": {
            "computed": True,
            "max_abs_delta_s": 1.8583333333333343,
        },
    }

    gate = evaluate_quality_report(report)

    assert gate["passed"] is True


def test_cli_summary_counts_failed_checks_and_warnings(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "annotation" / "quality_report.json"

    def fake_export_quality_report(**_kwargs):
        report_path.parent.mkdir(parents=True)
        report_path.write_text("{}", encoding="utf-8")
        return report_path

    def fake_evaluate_quality_report(_report, _rules):
        return {
            "passed": False,
            "risk_level": "high",
            "failed_checks": ["audio_video_duration"],
            "warnings": ["video_fps_delta"],
        }

    monkeypatch.setattr(cli, "export_quality_report", fake_export_quality_report)
    monkeypatch.setattr(cli, "evaluate_quality_report", fake_evaluate_quality_report)
    config = cli.RawHeadbandIntegrityConfig(
        session=tmp_path,
        output_root=tmp_path,
        camera_streams=("left",),
        start=0,
        num_frames=None,
        timestamp_tolerance_s=None,
        video_ext=".mkv",
        overwrite=True,
        environment_name="py312",
        repair_timestamp_files=True,
        max_repair_duplicate_rate=0.001,
        max_repair_consecutive_duplicate_frames=5,
        gate_enabled=True,
        gate_rules={},
    )

    summary = cli.run_raw_headband_integrity(config)

    assert summary["gate"]["blocking_check_count"] == 1
    assert summary["gate"]["warning_check_count"] == 1
