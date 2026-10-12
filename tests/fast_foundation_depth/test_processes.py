import pytest


def test_parent_exit_does_not_leave_sigterm_ignoring_descendant(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    from roboego_hand_vis.common import processes
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

    monkeypatch.setattr(processes.subprocess, "Popen", child)
    with pytest.raises(KeyboardInterrupt):
        processes.run_logged_command([sys.executable, "-c", "pass"], tmp_path / "stage.log", cwd=tmp_path)
    assert children[0].poll() is not None
    assert not processes.group_has_live_processes(children[0].pid)


def test_logged_nonzero_command_fails(tmp_path):
    import sys
    from roboego_hand_vis.common.processes import run_logged_command
    log = tmp_path / "stage.log"
    with pytest.raises(RuntimeError, match="code 7"):
        run_logged_command([sys.executable, "-c", "print(\"failure\"); raise SystemExit(7)"], log, cwd=tmp_path)
    assert "failure" in log.read_text()
