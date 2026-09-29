import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "hand_output", Path(__file__).parents[1] / "pipeline-output.py")
hand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hand)


def fake_process(code=0, artifacts=True):
    class Process:
        def __init__(self, command, stdout, **kwargs):
            final = Path(command[-2]) / "mint/video_canonical/final"
            final.mkdir(parents=True)
            if artifacts:
                (final / "hand_tracking.parquet").write_bytes(b"parquet")
                (final / "render.mp4").write_bytes(b"video")
            stdout.write("stage diagnostics\n")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return code
    return Process


@pytest.mark.parametrize("debug", [False, True])
def test_success_only_publishes_final_outputs(tmp_path, debug):
    with patch.object(hand, "revision", return_value={}), \
            patch.object(hand, "validate_artifacts", return_value={"frame_count": 100}), \
            patch.object(hand.subprocess, "Popen", fake_process()):
        assert hand.execute(["video.mp4", "depth", str(tmp_path), "100", *(["--debug"] if debug else [])]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {"hand_tracking.parquet", "render.mp4", "report.json"} | ({"debug"} if debug else set())
    assert json.loads((tmp_path / "report.json").read_text())["quality_status"] == "not_validated"


@pytest.mark.parametrize("code,artifacts", [(2, False), (0, False)])
def test_failure_keeps_diagnostics_without_final_artifacts(tmp_path, code, artifacts):
    with patch.object(hand, "revision", return_value={}), \
            patch.object(hand.subprocess, "Popen", fake_process(code, artifacts)):
        assert hand.execute(["video.mp4", "depth", str(tmp_path)]) != 0
    assert {p.name for p in tmp_path.iterdir()} == {"report.json", "error.log"}
    assert "stage diagnostics" in (tmp_path / "error.log").read_text()


def test_refuses_existing_results(tmp_path):
    (tmp_path / "report.json").write_text("previous")
    with pytest.raises(FileExistsError):
        hand.execute(["video.mp4", "depth", str(tmp_path)])
    assert (tmp_path / "report.json").read_text() == "previous"
