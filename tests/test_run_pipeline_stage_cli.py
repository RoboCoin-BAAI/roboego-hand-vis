from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from roboego_hand_vis.pipeline_launcher import WorkflowBlockedError
from scripts import run_pipeline_stage


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "session"
    (session / "videos").mkdir(parents=True)
    (session / "meta.json").write_text(json.dumps({"duration_sec": 60}), encoding="utf-8")
    return session


def test_run_pipeline_stage_records_success_by_default(monkeypatch, tmp_path: Path):
    session = _session(tmp_path)

    def fake_run(request, stage_callback=None):
        assert request.stage == "raw_quality"
        return ["kedro run --pipelines quality"]

    monkeypatch.setattr(run_pipeline_stage, "run_launcher_request", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_stage.py",
            "--dataset",
            "raw",
            "--session",
            str(session),
            "--stage",
            "raw_quality",
        ],
    )

    run_pipeline_stage.main()

    state = json.loads((session / "pipeline_state.json").read_text(encoding="utf-8"))
    record = state["stages"]["raw_quality"]
    assert record["status"] == "success"
    assert record["returncode"] == 0
    assert "run_pipeline_stage.py" in record["command"]


def test_run_pipeline_stage_records_workflow_steps_instead_of_workflow_alias(monkeypatch, tmp_path: Path):
    session = _session(tmp_path)

    def fake_run(request, stage_callback=None):
        for stage in [
            "qwen_annotation",
            "post_annotation_actions",
            "processing_plan",
            "hand_windows",
        ]:
            stage_callback(stage, "processing", [], None)
            stage_callback(stage, "success", [], None)
        return ["kedro qwen", "kedro post", "kedro plan", "kedro windows"]

    monkeypatch.setattr(run_pipeline_stage, "run_launcher_request", fake_run)
    monkeypatch.setattr(run_pipeline_stage, "workflow_stage_names", lambda *args, **kwargs: [
        "qwen_annotation",
        "post_annotation_actions",
        "processing_plan",
        "hand_windows",
    ])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_stage.py",
            "--dataset",
            "raw",
            "--session",
            str(session),
            "--stage",
            "base_prepare",
        ],
    )

    run_pipeline_stage.main()

    state = json.loads((session / "pipeline_state.json").read_text(encoding="utf-8"))
    assert "base_prepare" not in state["stages"]
    assert state["stages"]["qwen_annotation"]["status"] == "success"
    assert state["stages"]["post_annotation_actions"]["status"] == "success"
    assert state["stages"]["processing_plan"]["status"] == "success"
    assert state["stages"]["hand_windows"]["status"] == "success"


def test_run_pipeline_stage_records_blocked_gate_on_workflow_steps(monkeypatch, tmp_path: Path):
    session = _session(tmp_path)

    def fake_run(request, stage_callback=None):
        stage_callback("qwen_annotation", "processing", [], None)
        stage_callback("qwen_annotation", "blocked", [], WorkflowBlockedError("RAW QUALITY GATE FAILED\nimu_jump"))
        raise WorkflowBlockedError("RAW QUALITY GATE FAILED\nimu_jump")

    monkeypatch.setattr(run_pipeline_stage, "run_launcher_request", fake_run)
    monkeypatch.setattr(run_pipeline_stage, "workflow_stage_names", lambda *args, **kwargs: [
        "qwen_annotation",
        "post_annotation_actions",
        "processing_plan",
        "hand_windows",
    ])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_stage.py",
            "--dataset",
            "raw",
            "--session",
            str(session),
            "--stage",
            "base_prepare",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_pipeline_stage.main()

    state = json.loads((session / "pipeline_state.json").read_text(encoding="utf-8"))
    assert "base_prepare" not in state["stages"]
    record = state["stages"]["qwen_annotation"]
    assert exc_info.value.code == 2
    assert record["status"] == "blocked"
    assert record["returncode"] == 2
    assert "imu_jump" in record["error"]


def test_run_pipeline_stage_dry_run_does_not_write_state(monkeypatch, tmp_path: Path):
    session = _session(tmp_path)

    def fake_run(request, stage_callback=None):
        assert request.dry_run is True
        return ["kedro run --dry-run"]

    monkeypatch.setattr(run_pipeline_stage, "run_launcher_request", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_stage.py",
            "--dataset",
            "raw",
            "--session",
            str(session),
            "--stage",
            "raw_quality",
            "--dry-run",
        ],
    )

    run_pipeline_stage.main()

    assert not (session / "pipeline_state.json").exists()
