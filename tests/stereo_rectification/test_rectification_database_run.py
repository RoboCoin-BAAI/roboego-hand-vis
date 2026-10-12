import pytest
import yaml

from roboego_hand_vis.stereo_rectification.database_run import run_pending


class Service:
    def __init__(self, paths):
        self.rows = [dict(id=i, local_path=path) for i, path in enumerate(paths)]
        self.results = []

    def claim_stereo_rectification(self):
        return self.rows.pop(0) if self.rows else None

    def finish_stereo_rectification(self, data_id, *, succeeded):
        self.results.append((data_id, succeeded))
        return True


def options(**changes):
    return dict(dataset_adapter="raw", model="equidistant", balance=0, width=1600,
                height=900, overwrite=False, dry_run=False) | changes


def session(root, name, model="pinhole"):
    path = root / name
    calib = path / "calibrations"
    calib.mkdir(parents=True)
    info = dict(camera_model=model, distortion_model="equidistant",
                intrinsics=[550, 550, 960, 540], distortion_coeffs=[0.1, 0, 0, 0])
    (calib / "camera-camchain.yaml").write_text(yaml.safe_dump({"cam0": info, "cam1": info}))
    return path


def publish(path):
    for rel in ("videos/left_rectified.mp4", "videos/right_rectified.mp4", "dataset_manifest/dataset_manifest.json"):
        target = path / rel
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(b"output")


def test_success_and_execution_failure(tmp_path):
    for name in ("ok", "error"):
        session(tmp_path, name)
    service = Service(["ok", "error"])

    def check(path, opts):
        assert path == tmp_path / path.name
        assert opts == options()
        if path.name == "error":
            raise RuntimeError("rectification failed")
        publish(path)
        return {"status": "ok"}

    result = run_pending(service, tmp_path, options(), runner=check)
    assert [row["status"] for row in result] == ["completed", "failed"]
    assert service.results == [(0, True), (1, False)]


def test_unsupported_model_and_unsafe_paths_do_not_run(tmp_path):
    session(tmp_path, "ds", "ds")
    service = Service(["ds", "../outside", "missing"])
    result = run_pending(service, tmp_path, options(), runner=lambda *a: pytest.fail("must not run"))
    assert all(row["status"] == "failed" for row in result)
    assert "双球面" in result[0]["error"]
    assert service.results == [(0, False), (1, False), (2, False)]


def test_missing_outputs_cannot_complete(tmp_path):
    session(tmp_path, "session")
    service = Service(["session"])
    result = run_pending(service, tmp_path, options(), runner=lambda *a: {"status": "ok"})
    assert result[0]["status"] == "failed"
    assert service.results == [(0, False)]


def test_dry_run_cannot_claim_or_complete(tmp_path):
    service = Service(["session"])
    with pytest.raises(ValueError, match="dry_run"):
        run_pending(service, tmp_path, options(dry_run=True))
    assert len(service.rows) == 1


def test_limit_interruption_and_writeback(tmp_path):
    session(tmp_path, "session")
    service = Service(["session", "session"])

    def check(path, opts):
        publish(path)
        return {"status": "ok"}

    run_pending(service, tmp_path, options(), limit=1, runner=check)
    assert len(service.rows) == 1

    def interrupted(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pending(service, tmp_path, options(), runner=interrupted)
    assert service.results == [(0, True), (1, False)]
    service = Service(["session", "session"])
    service.finish_stereo_rectification = lambda *a, **kw: False
    with pytest.raises(RuntimeError, match="write"):
        run_pending(service, tmp_path, options(), runner=check)
    assert len(service.rows) == 1


@pytest.mark.parametrize("code", [0, 2])
def test_child_command_and_log(tmp_path, monkeypatch, code):
    import subprocess
    import sys
    from roboego_hand_vis.stereo_rectification import database_run
    original = subprocess.Popen
    commands = []

    def child(command, **kwargs):
        commands.append(command)
        return original([sys.executable, "-c", f"print('rectifier output'); raise SystemExit({code})"], **kwargs)

    monkeypatch.setattr(database_run.subprocess, "Popen", child)
    if code:
        with pytest.raises(RuntimeError, match="code 2"):
            database_run._run_session(tmp_path, options())
    else:
        assert database_run._run_session(tmp_path, options())["status"] == "ok"
    assert "--no-dry_run" in commands[0]
    assert "--no-overwrite" in commands[0]
    assert commands[0][2] == "roboego_hand_vis.stereo_rectification.run"
    assert "rectifier output" in (tmp_path / "annotation" / "stereo_rectification.log").read_text()


def test_interruption_stops_real_child_before_return(tmp_path, monkeypatch):
    import subprocess
    import sys
    from roboego_hand_vis.stereo_rectification import database_run
    original = subprocess.Popen
    children = []

    def child(command, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(process)
        original_wait = process.wait
        first = True

        def interrupted_wait(*args, **kw):
            nonlocal first
            if first:
                first = False
                raise KeyboardInterrupt
            return original_wait(*args, **kw)

        process.wait = interrupted_wait
        return process

    monkeypatch.setattr(database_run.subprocess, "Popen", child)
    with pytest.raises(KeyboardInterrupt):
        database_run._run_session(tmp_path, options())
    assert children[0].poll() is not None


def test_parent_exit_does_not_leave_sigterm_ignoring_descendant(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    from roboego_hand_vis.stereo_rectification import database_run
    original = subprocess.Popen
    marker = tmp_path / "grandchild.pid"
    children = []
    grandchild_code = ("import os, signal, time; from pathlib import Path; "
                      "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                      f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)")
    parent_code = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {grandchild_code!r}]); time.sleep(60)"

    def child(command, **kwargs):
        process = original([sys.executable, "-c", parent_code], **kwargs)
        children.append(process)
        original_wait = process.wait
        first = True

        def interrupted_wait(*args, **kw):
            nonlocal first
            if first:
                first = False
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert marker.exists()
                raise KeyboardInterrupt
            return original_wait(*args, **kw)

        process.wait = interrupted_wait
        return process

    monkeypatch.setattr(database_run.subprocess, "Popen", child)
    with pytest.raises(KeyboardInterrupt):
        database_run._run_session(tmp_path, options())
    assert children[0].poll() is not None
    assert not database_run._group_has_live_processes(children[0].pid)
