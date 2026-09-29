import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("runtime_output", Path(__file__).parents[1] / "runtime-output.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def fake_process(code=0, artifact=True):
    class Process:
        def __init__(self, command, stdout, **kwargs):
            root = Path(command[command.index("--output_dir") + 1]) / "trajectory"
            root.mkdir(parents=True)
            (root / "run_report.json").write_text(json.dumps({"status": "complete" if code == 0 else "not_initialized",
                                                              "pose_count": 2 if artifact else 0}))
            if artifact:
                np.savez(root / "trajectory.npz", timestamp_camera=[0., 1.])
            stdout.write("diagnostic output\n")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return code
    return Process


@pytest.mark.parametrize("debug", [False, True])
def test_success_publishes_only_contract(tmp_path, debug):
    with patch.object(runtime.subprocess, "Popen", fake_process()):
        assert runtime.execute(["--output_dir", str(tmp_path), *(["--debug"] if debug else [])]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {"trajectory.npz", "report.json"} | ({"debug"} if debug else set())
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["quality_status"] == "not_validated"
    assert report["artifacts"]["trajectory"] == "trajectory.npz"


@pytest.mark.parametrize("code,artifact", [(3, False), (0, False)])
def test_failure_preserves_report_and_log_not_trajectory(tmp_path, code, artifact):
    with patch.object(runtime.subprocess, "Popen", fake_process(code, artifact)):
        assert runtime.execute(["--output_dir", str(tmp_path)]) != 0
    assert {p.name for p in tmp_path.iterdir()} == {"report.json", "error.log"}
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "failed"
    assert "diagnostic output" in (tmp_path / "error.log").read_text()


def test_existing_output_untouched(tmp_path):
    (tmp_path / "report.json").write_text("original")
    with patch.object(runtime.subprocess, "Popen") as process:
        with pytest.raises(FileExistsError):
            runtime.execute(["--output_dir", str(tmp_path)])
        process.assert_not_called()
    assert (tmp_path / "report.json").read_text() == "original"
