from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_missing_fast_depth_windows import (
    build_depth_command,
    find_missing_depth_windows,
    patch_depth_usable_report,
    run_missing_depth_windows,
)


def make_session(root: Path) -> Path:
    session = root / "cp_0001/session_20260826_000000/divide/session_task"
    (session / "videos").mkdir(parents=True)
    (session / "meta.json").write_text("{}", encoding="utf-8")
    (session / "pipeline_state.json").write_text(
        json.dumps(
            {
                "stages": {
                    "prepare_stereo": {"status": "success"},
                    "stereo_summary_quality": {"status": "success"},
                    "fast_foundation_depth": {"status": "blocked"},
                }
            }
        ),
        encoding="utf-8",
    )
    return session


def make_hand_window(session: Path, name: str, *, complete_depth: bool, usable: bool = False) -> Path:
    unit = session / "hand_windows" / name
    (unit / "dataset_manifest").mkdir(parents=True)
    (unit / "dataset_manifest/dataset_manifest.json").write_text("{}", encoding="utf-8")
    report = unit / "quality/stereo_summary/stereo_quality_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "passed": False,
                    "usable_for_depth": usable,
                    "depth_risk_level": "blocked" if not usable else "high_risk",
                    "depth_risk_reasons": ["geometry_too_poor"],
                },
                "next_step_gate_zh": {
                    "usable_for_depth": usable,
                    "depth_risk_level": "blocked" if not usable else "high_risk",
                },
            }
        ),
        encoding="utf-8",
    )
    if complete_depth:
        depth_dir = unit / "depth/fast_foundation/depth_uint16_png"
        depth_dir.mkdir(parents=True)
        (unit / "depth/fast_foundation/fast_foundation_stereo_video_meta.json").write_text("{}", encoding="utf-8")
        (depth_dir / "frame_000000_depth_mm.png").write_bytes(b"png")
    return unit


def test_find_missing_depth_windows_selects_blocked_sessions_with_missing_unit_depth(tmp_path):
    session = make_session(tmp_path)
    missing = make_hand_window(session, "handwin_0000", complete_depth=False)
    make_hand_window(session, "handwin_0001", complete_depth=True)

    windows = find_missing_depth_windows(tmp_path)

    assert [item.unit for item in windows] == [missing.resolve()]


def test_patch_depth_usable_report_updates_summary_and_next_gate(tmp_path):
    session = make_session(tmp_path)
    unit = make_hand_window(session, "handwin_0000", complete_depth=False)

    changed = patch_depth_usable_report(unit)

    report = json.loads((unit / "quality/stereo_summary/stereo_quality_report.json").read_text(encoding="utf-8"))
    assert changed is True
    assert report["summary"]["usable_for_depth"] is True
    assert report["summary"]["depth_risk_level"] == "high_risk"
    assert report["next_step_gate_zh"]["usable_for_depth"] is True
    assert report["next_step_gate_zh"]["depth_risk_level"] == "high_risk"


def test_run_missing_depth_windows_patches_and_runs_hand_window_commands(monkeypatch, tmp_path):
    session = make_session(tmp_path)
    unit = make_hand_window(session, "handwin_0000", complete_depth=False)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("scripts.run_missing_fast_depth_windows.subprocess.run", fake_run)

    result = run_missing_depth_windows(
        root=tmp_path,
        dataset="raw",
        python_bin=sys.executable,
        config=None,
        dry_run=False,
        patch_reports=True,
        limit=0,
        exit_on_error=False,
    )

    assert result["status_counts"] == {"success": 1}
    assert calls == [
        build_depth_command(
            python_bin=sys.executable,
            dataset="raw",
            session=session.resolve(),
            hand_window=unit.resolve(),
            config=None,
        )
    ]
