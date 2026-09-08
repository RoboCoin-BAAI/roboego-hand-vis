from __future__ import annotations

import json
import subprocess
from pathlib import Path

from roboego_hand_vis.pipeline_state import (
    discover_sessions,
    print_state_report,
    read_pipeline_state,
    render_dashboard,
    run_batch_stage,
    stage_artifact_outcome,
    update_stage_state,
)


def _make_session(root: Path, name: str) -> Path:
    session = root / name
    (session / "videos").mkdir(parents=True)
    (session / "videos" / "left.mp4").write_bytes(b"")
    (session / "meta.json").write_text(json.dumps({"duration_sec": 60}), encoding="utf-8")
    return session


def test_update_stage_state_writes_session_pipeline_state(tmp_path: Path):
    session = _make_session(tmp_path, "session_a")

    state = update_stage_state(
        session,
        "raw_quality",
        "success",
        command="python scripts/run_pipeline_stage.py --stage raw_quality",
        duration_s=1.25,
        outputs=["annotation/quality_report.json"],
    )

    state_path = session / "pipeline_state.json"
    written = json.loads(state_path.read_text(encoding="utf-8"))

    assert state["stages"]["raw_quality"]["status"] == "success"
    assert written["session"] == str(session.resolve())
    assert written["stages"]["raw_quality"]["duration_s"] == 1.25
    assert written["stages"]["raw_quality"]["outputs"] == ["annotation/quality_report.json"]
    assert written["stages"]["raw_quality"]["error"] is None


def test_discover_sessions_finds_direct_and_nested_sessions(tmp_path: Path):
    direct = _make_session(tmp_path, "session_direct")
    nested = _make_session(tmp_path / "cp_0001", "session_nested")
    _make_session(direct / "hand_windows", "handwin_0000_t000000_000065")
    (tmp_path / "notes").mkdir()

    sessions = discover_sessions(tmp_path)

    assert sessions == [nested.resolve(), direct.resolve()]


def test_discover_sessions_uses_fast_baai_divide_layout_without_rglob(monkeypatch, tmp_path: Path):
    session = _make_session(tmp_path / "cp_0001/session_20260826_100651/divide", "session_20260826_100651_task")

    def fail_rglob(self, pattern):  # noqa: ANN001
        raise AssertionError("rglob should not be needed for baai divide roots")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    assert discover_sessions(tmp_path) == [session.resolve()]


def test_run_batch_stage_marks_success_and_writes_dashboard(monkeypatch, tmp_path: Path):
    session = _make_session(tmp_path, "session_ok")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr("roboego_hand_vis.pipeline_state.subprocess.run", fake_run)

    summary = run_batch_stage(
        root=tmp_path,
        dataset="raw",
        stage="raw_quality",
        workers=1,
        dashboard=True,
        dashboard_path=tmp_path / "dashboard.html",
    )

    state = json.loads((session / "pipeline_state.json").read_text(encoding="utf-8"))
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert summary["status_counts"] == {"success": 1}
    assert "--stage" in calls[0]
    assert "raw_quality" in calls[0]
    assert state["stages"]["raw_quality"]["status"] == "success"
    assert "session_ok" in html
    assert "raw_quality" in html
    assert "success" in html


def test_run_batch_stage_marks_raw_quality_blocked_when_gate_report_failed(monkeypatch, tmp_path: Path):
    session = _make_session(tmp_path, "session_raw_gate_failed")
    report_path = session / "annotation/quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "raw_headband_quality_gate": {
                    "passed": False,
                    "status": "failed",
                    "failed_checks": ["timestamp_duplicates:left"],
                    "warnings": [],
                    "checks": [
                        {
                            "name": "timestamp_duplicates",
                            "stream": "left",
                            "message": "left timestamp duplicate rate is 0.004",
                            "value": 0.004,
                            "threshold": 0.001,
                            "passed": False,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="kedro ok\n", stderr="")

    monkeypatch.setattr("roboego_hand_vis.pipeline_state.subprocess.run", fake_run)

    summary = run_batch_stage(root=tmp_path, dataset="raw", stage="raw_quality", workers=1)

    state = json.loads((session / "pipeline_state.json").read_text(encoding="utf-8"))
    stage_state = state["stages"]["raw_quality"]

    assert summary["status_counts"] == {"blocked": 1}
    assert stage_state["status"] == "blocked"
    assert stage_state["returncode"] == 2
    assert "timestamp_duplicates:left" in stage_state["error"]


def test_run_batch_stage_force_passes_force_to_single_stage_command(monkeypatch, tmp_path: Path):
    _make_session(tmp_path, "session_force")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr("roboego_hand_vis.pipeline_state.subprocess.run", fake_run)

    run_batch_stage(root=tmp_path, dataset="raw", stage="base_prepare", workers=1, force=True)

    assert "--force" in calls[0]


def test_run_batch_stage_marks_blocked_for_gate_exit_code(monkeypatch, tmp_path: Path):
    session = _make_session(tmp_path, "session_blocked")

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr="RAW QUALITY GATE FAILED\nimu_jump\n")

    monkeypatch.setattr("roboego_hand_vis.pipeline_state.subprocess.run", fake_run)

    summary = run_batch_stage(root=tmp_path, dataset="raw", stage="raw_quality", workers=1)

    state = json.loads((session / "pipeline_state.json").read_text(encoding="utf-8"))
    stage_state = state["stages"]["raw_quality"]

    assert summary["status_counts"] == {"blocked": 1}
    assert stage_state["status"] == "blocked"
    assert "imu_jump" in stage_state["error"]


def test_fast_foundation_depth_artifact_outcome_blocks_when_outputs_missing(tmp_path: Path):
    session = _make_session(tmp_path, "session_depth_missing")
    manifest = session / "dataset_manifest/dataset_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"videos": {"left": {"frame_count": 30}}}), encoding="utf-8")

    outcome = stage_artifact_outcome(session, "fast_foundation_depth")

    assert outcome is not None
    assert outcome["status"] == "blocked"
    assert outcome["returncode"] == 2
    assert "FAST FOUNDATION DEPTH OUTPUT MISSING" in outcome["error"]
    assert "stereo_summary_quality" in outcome["error"]


def test_fast_foundation_depth_artifact_outcome_passes_when_meta_exists(tmp_path: Path):
    session = _make_session(tmp_path, "session_depth_done")
    manifest = session / "dataset_manifest/dataset_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"videos": {"left": {"frame_count": 30}}}), encoding="utf-8")
    depth_meta = session / "depth/fast_foundation/fast_foundation_stereo_video_meta.json"
    depth_meta.parent.mkdir(parents=True)
    depth_meta.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    assert stage_artifact_outcome(session, "fast_foundation_depth") is None


def test_archive_depth_cache_artifact_outcome_blocks_until_depth_stage_success(tmp_path: Path):
    session = _make_session(tmp_path, "session_archive_depth_processing")
    update_stage_state(session, "fast_foundation_depth", "processing")

    outcome = stage_artifact_outcome(session, "archive_depth_cache")

    assert outcome is not None
    assert outcome["status"] == "blocked"
    assert outcome["returncode"] == 2
    assert "required_depth_stage=fast_foundation_depth" in outcome["error"]
    assert "depth_stage_status=processing" in outcome["error"]


def test_archive_depth_cache_artifact_outcome_passes_after_depth_stage_success(tmp_path: Path):
    session = _make_session(tmp_path, "session_archive_depth_done")
    update_stage_state(session, "fast_foundation_depth", "success")

    assert stage_artifact_outcome(session, "archive_depth_cache") is None


def test_depth_estimation_workflow_alias_expands_to_dashboard_stages(tmp_path: Path):
    session = _make_session(tmp_path, "session_depth_estimation")
    update_stage_state(session, "depth_estimation", "success", duration_s=2.0)

    state = read_pipeline_state(session)

    assert state["stages"]["stereo_summary_quality"]["status"] == "success"
    assert state["stages"]["stereo_summary_quality"]["derived_from"] == "depth_estimation"
    assert state["stages"]["fast_foundation_depth"]["status"] == "success"
    assert state["stages"]["fast_foundation_depth"]["derived_from"] == "depth_estimation"


def test_run_batch_stage_dry_run_does_not_write_pipeline_state(tmp_path: Path):
    session = _make_session(tmp_path, "session_dry")

    summary = run_batch_stage(root=tmp_path, dataset="raw", stage="raw_quality", workers=1, dry_run=True)

    assert summary["status_counts"] == {"skipped": 1}
    assert not (session / "pipeline_state.json").exists()


def test_render_dashboard_shows_all_stage_statuses(tmp_path: Path):
    session = _make_session(tmp_path, "session_dash")
    update_stage_state(session, "base_prepare", "success", duration_s=2.0)
    update_stage_state(session, "raw_quality", "success", duration_s=1.0)
    update_stage_state(session, "video_simple_quality", "fail", error="decode failed")

    output = render_dashboard([session], tmp_path / "index.html")
    html = output.read_text(encoding="utf-8")

    assert "Pipeline Dashboard" in html
    assert "session_dash" in html
    assert "<th>base_prepare</th>" not in html
    assert "raw_quality" in html
    assert "video_simple_quality" in html
    assert "decode failed" in html


def test_render_dashboard_adds_quality_gate_columns_from_reports(tmp_path: Path):
    session = _make_session(tmp_path, "session_quality_gate")
    update_stage_state(session, "raw_quality", "success", duration_s=1.0)
    update_stage_state(session, "video_simple_quality", "success", duration_s=2.0)
    quality_report = session / "annotation/quality_report.json"
    video_report = session / "annotation/video_simple_quality_left.json"
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_report.write_text(
        json.dumps(
            {
                "raw_headband_quality_gate": {
                    "passed": False,
                    "status": "failed",
                    "failed_checks": ["frame_count_consistency"],
                    "checks": [
                        {
                            "name": "frame_count_consistency",
                            "passed": False,
                            "message": "Max frame count delta across videos/timestamps is 5",
                            "value": 5,
                            "threshold": 0,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    video_report.write_text(json.dumps({"passed": True, "status": "passed"}), encoding="utf-8")

    output = render_dashboard([session], tmp_path / "index.html")
    html = output.read_text(encoding="utf-8")

    assert "<th>raw_quality_gate</th>" in html
    assert "<th>video_simple_quality_gate</th>" in html
    assert 'title="raw_quality_gate | RAW QUALITY GATE FAILED' in html
    assert ">blocked</td>" in html
    assert 'title="video_simple_quality_gate' in html
    assert ">success</td>" in html


def test_print_state_report_counts_gate_conclusions_without_pending_noise(tmp_path: Path, capsys):
    session = _make_session(tmp_path, "session_quality_gate")
    update_stage_state(session, "raw_quality", "success", duration_s=1.0)
    update_stage_state(session, "video_simple_quality", "success", duration_s=2.0)
    quality_report = session / "annotation/quality_report.json"
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_report.write_text(
        json.dumps(
            {
                "raw_headband_quality_gate": {
                    "passed": False,
                    "status": "failed",
                    "failed_checks": ["frame_count_consistency"],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = print_state_report(tmp_path)
    captured = capsys.readouterr()

    assert payload["stage_counts"]["raw_quality"] == {"success": 1}
    assert payload["stage_counts"]["raw_quality_gate"] == {"blocked": 1}
    assert "pending" not in captured.out


def test_render_dashboard_expands_base_prepare_workflow_statuses(tmp_path: Path):
    session = _make_session(tmp_path, "session_base_prepare")
    update_stage_state(session, "base_prepare", "success", duration_s=9.0)

    output = render_dashboard([session], tmp_path / "index.html")
    html = output.read_text(encoding="utf-8")

    assert "<th>base_prepare</th>" not in html
    assert 'title="raw_quality | duration_s=9.0 | derived_from=base_prepare"' in html
    assert 'title="qwen_annotation | duration_s=9.0 | derived_from=base_prepare"' in html
    assert 'title="hand_windows | duration_s=9.0 | derived_from=base_prepare"' in html
    assert 'title="prepare_stereo | duration_s=9.0 | derived_from=base_prepare"' in html
