"""Linux process-group cleanup for long-running stage commands."""

import os
import signal
import subprocess
import time
from pathlib import Path


def group_has_live_processes(group_id):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group_id and fields[0] not in ("Z", "X"):
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False


def stop_process_group(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    process.wait()
    while group_has_live_processes(process.pid):
        time.sleep(0.02)


def run_logged_command(command, log_path, *, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait()
        except BaseException:
            stop_process_group(process)
            raise
    if group_has_live_processes(process.pid):
        stop_process_group(process)
    if code:
        raise RuntimeError(f"Stage exited with code {code}; see {log_path}")
