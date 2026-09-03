from __future__ import annotations

import json
import subprocess
from pathlib import Path

from roboego_hand_vis.pipeline_state import (
    discover_sessions,
    render_dashboard,
    run_batch_stage,
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
