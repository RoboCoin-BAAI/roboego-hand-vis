from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.batch_run_hand_depth_final import (
    FINAL_HAND_DEPTH_STAGES,
    build_stage_command,
    run_one_session,
)


def make_session(root: Path, name: str) -> Path:
    session = root / name
    (session / "videos").mkdir(parents=True)
    (session / "meta.json").write_text("{}", encoding="utf-8")
    return session


def write_state(session: Path, stages: dict[str, str]) -> None:
    (session / "pipeline_state.json").write_text(
        json.dumps({"stages": {stage: {"status": status} for stage, status in stages.items()}}),
        encoding="utf-8",
    )


def test_build_stage_command_uses_requested_python_dataset_session_and_stage(tmp_path):
    session = make_session(tmp_path, "session_a")

    cmd = build_stage_command(
        python_bin="/opt/ego/bin/python",
        dataset="raw",
        session=session,
        stage="hand_filter",
        config=Path("conf/datasets/raw.yml"),
        force=True,
    )

    assert cmd == [
        "/opt/ego/bin/python",
        "scripts/run_pipeline_stage.py",
        "--dataset",
        "raw",
        "--session",
        str(session.resolve()),
        "--stage",
        "hand_filter",
        "--config",
        "conf/datasets/raw.yml",
        "--force",
    ]


def test_run_one_session_skips_without_successful_hamer_and_depth(tmp_path):
    session = make_session(tmp_path, "session_missing_depth")
    write_state(session, {"hamer": "success", "fast_foundation_depth": "blocked"})

    result = run_one_session(
        session=session,
        dataset="raw",
        python_bin=sys.executable,
        config=None,
        force=False,
        dry_run=True,
        exit_on_error=False,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "required_stage_not_success"
    assert result["required_statuses"]["hamer"] == "success"
    assert result["required_statuses"]["fast_foundation_depth"] == "blocked"


def test_run_one_session_runs_remaining_stages_and_skips_success(monkeypatch, tmp_path):
    session = make_session(tmp_path, "session_ready")
    write_state(session, {"hamer": "success", "fast_foundation_depth": "success", "hand_filter": "success"})
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.batch_run_hand_depth_final.subprocess.run", fake_run)

    result = run_one_session(
        session=session,
        dataset="raw",
        python_bin=sys.executable,
        config=None,
        force=False,
        dry_run=False,
        exit_on_error=False,
    )

    assert result["status"] == "success"
    assert result["stage_counts"]["skipped"] == 1
    assert result["stage_counts"]["success"] == len(FINAL_HAND_DEPTH_STAGES) - 1
    assert [cmd[cmd.index("--stage") + 1] for cmd in calls] == list(FINAL_HAND_DEPTH_STAGES[1:])
