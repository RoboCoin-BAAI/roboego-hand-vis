import json
import sys

from roboego_hand_vis.core.reports import write_stage_report
from roboego_hand_vis.core.runtime import PythonRuntime


def test_python_runtime_builds_command_with_configured_interpreter():
    runtime = PythonRuntime(name="ego", python=sys.executable)

    assert runtime.command(["-c", "print('ok')"]) == [
        sys.executable,
        "-c",
        "print('ok')",
    ]


def test_write_stage_report_persists_stage_metadata(tmp_path):
    report_path = tmp_path / "reports" / "depth.json"

    report = write_stage_report(
        report_path,
        stage="depth",
        status="complete",
        metrics={"frames": 15},
        artifacts={"vis": "processed/foundation_depth/vis"},
        params={"scale": 1.0},
        message="smoke",
    )

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == report
    assert saved["stage"] == "depth"
    assert saved["status"] == "complete"
    assert saved["metrics"] == {"frames": 15}
    assert saved["artifacts"] == {"vis": "processed/foundation_depth/vis"}
    assert saved["params"] == {"scale": 1.0}
    assert saved["message"] == "smoke"
    assert "created_at" in saved
