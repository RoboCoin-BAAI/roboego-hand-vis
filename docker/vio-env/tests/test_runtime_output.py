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
            valid = np.array([False, code == 0, code == 0])
            report = {"status": "complete" if code == 0 else "not_initialized",
                      "pose_count": int(valid.sum()) if artifact else 0}
            if artifact:
                report['frame_alignment'] = dict(frame_count=3, valid_pose_count=int(valid.sum()))
                matrices = np.full((3, 4, 4), np.nan)
                matrices[valid] = np.eye(4)
                np.savez(root / "trajectory.npz", schema_version='openvins.frame_aligned.v2',
                         timestamp_camera=[0., 1., 2.], frame_index=np.arange(3),
                         status=np.where(valid, 'valid', 'initializing'),
                         T_world_camera=matrices, T_world_imu=matrices)
            (root / "run_report.json").write_text(json.dumps(report))
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


def test_uninitialized_run_delivers_invalid_rows_but_still_fails(tmp_path):
    with patch.object(runtime.subprocess, 'Popen', fake_process(code=3)):
        assert runtime.execute(['--output_dir', str(tmp_path)]) == 3
    assert {p.name for p in tmp_path.iterdir()} == {'trajectory.npz', 'report.json', 'error.log'}
    with np.load(tmp_path / 'trajectory.npz', allow_pickle=False) as data:
        assert len(data['status']) == 3
        assert not (data['status'] == 'valid').any()
        assert not {'valid', 'processed', 'initialized'}.intersection(data.files)
        assert np.isnan(data['T_world_camera']).all()
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['status'] == 'failed'
    assert report['execution_status'] == 'not_initialized'


def test_invalid_matrix_is_not_published(tmp_path):
    original = fake_process()
    class BadProcess(original):
        def __init__(self, command, stdout, **kwargs):
            super().__init__(command, stdout, **kwargs)
            path = Path(command[command.index('--output_dir') + 1]) / 'trajectory/trajectory.npz'
            with np.load(path) as data:
                arrays = dict(data)
            arrays['T_world_camera'][0] = np.eye(4)
            np.savez(path, **arrays)
    with patch.object(runtime.subprocess, 'Popen', BadProcess):
        assert runtime.execute(['--output_dir', str(tmp_path)]) != 0
    assert not (tmp_path / 'trajectory.npz').exists()


def test_time_offset_mode_reaches_openvins_without_changing_project_cli(tmp_path):
    with patch.object(runtime.subprocess, "Popen", side_effect=fake_process()) as process:
        assert runtime.execute(["--output_dir", str(tmp_path), "--time-offset-mode", "config"]) == 0
    assert process.call_args.kwargs["env"]["OPENVINS_TIME_OFFSET_MODE"] == "config"
    assert "--time-offset-mode" not in process.call_args.args[0]
    assert json.loads((tmp_path / "report.json").read_text())["time_offset_mode"] == "config"


def test_timing_bootstrap_mode_reaches_openvins_without_changing_project_cli(tmp_path):
    with patch.object(runtime.subprocess, "Popen", side_effect=fake_process()) as process:
        assert runtime.execute(["--output_dir", str(tmp_path), "--timing-bootstrap", "off"]) == 0
    assert process.call_args.kwargs["env"]["OPENVINS_TIMING_BOOTSTRAP"] == "off"
    assert "--timing-bootstrap" not in process.call_args.args[0]
    assert json.loads((tmp_path / "report.json").read_text())["timing_bootstrap_mode"] == "off"
