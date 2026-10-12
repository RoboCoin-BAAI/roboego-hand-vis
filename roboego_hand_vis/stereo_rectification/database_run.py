"""L1 queue wrapper around the existing session rectification command."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

from .compute import (DEFAULT_CONFIG, PROJECT_ROOT, _validate_calibration,
                      load_yaml, resolved_options_from_yaml_and_args)


OUTPUTS = ("videos/left_rectified.mp4", "videos/right_rectified.mp4",
           "dataset_manifest/dataset_manifest.json")


def _group_has_live_processes(group_id):
    """Linux: zombie members cannot write outputs and need not delay cleanup."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm may contain spaces/parentheses; fields after the final ')' are
            # state, parent PID, process group, ...
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group_id and fields[0] not in ("Z", "X"):
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False


def _stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The parent may already be gone while an encoder ignores SIGTERM. Always
    # kill remaining group members, rather than keying cleanup off parent exit.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    while _group_has_live_processes(process.pid):
        time.sleep(0.02)


def _validate(shared_root, options, limit):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if options["dry_run"]:
        raise ValueError("Database mode does not support dry_run: no outputs would be generated")
    root = Path(shared_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Shared root is not a directory: {root}")
    return root


def _run_session(session, options):
    command = [sys.executable, "-m", "roboego_hand_vis.stereo_rectification.run",
               "--session", str(session), "--no-dry_run",
               "--overwrite" if options["overwrite"] else "--no-overwrite"]
    for key in ("dataset_adapter", "model", "balance", "width", "height"):
        command.extend([f"--{key}", str(options[key])])
    log = session / "annotation" / "stereo_rectification.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
        except BaseException:
            # Stop the original CLI and its rectifier/encoder children before
            # returning the database task to a terminal state.
            _stop_process_group(process)
            raise
    if _group_has_live_processes(process.pid):
        _stop_process_group(process)
    if code:
        raise RuntimeError(f"Rectification exited with code {code}; see {log}")
    return {"status": "ok", "log_path": str(log)}


def run_pending(service, shared_root, options, *, limit=0, runner=None) -> list[dict]:
    """Claim in short transactions; rectify and publish before completing."""
    root = _validate(shared_root, options, limit)
    runner = runner or _run_session
    results = []
    while not limit or len(results) < limit:
        row = service.claim_stereo_rectification()
        if row is None:
            break
        result = {"id": row["id"], "status": "failed"}
        succeeded = False
        try:
            local = row["local_path"]
            path = Path(local)
            if (not local.strip() or path.is_absolute() or ".." in path.parts
                    or "\\" in local or PureWindowsPath(local).drive or str(path) == "."):
                raise ValueError("local_path must be relative to the shared root")
            session = (root / path).resolve()
            if not session.is_relative_to(root) or not session.is_dir():
                raise ValueError(f"Session is missing or outside the shared root: {local}")
            _validate_calibration(session, session, options["model"])
            summary = runner(session, options)
            if summary.get("status") != "ok":
                raise RuntimeError("Rectification did not report successful execution")
            for rel in OUTPUTS:
                output = (session / rel).resolve()
                if not output.is_relative_to(root) or not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError(f"Missing or invalid rectification output: {rel}")
            result["summary"] = summary | {"outputs": [str(session / rel) for rel in OUTPUTS]}
            result["status"] = "completed"
            succeeded = True
        except Exception as exc:
            result["error"] = str(exc)
        except BaseException:
            if not service.finish_stereo_rectification(row["id"], succeeded=False):
                raise RuntimeError(f"Failed to write interrupted task status: {row['id']}")
            raise
        if not service.finish_stereo_rectification(row["id"], succeeded=succeeded):
            raise RuntimeError(f"Failed to write rectification status: {row['id']}")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="从 L1 数据库领取视频质检已通过的数据，进行去畸变和立体矫正。")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--limit", type=int, default=0, help="最多处理条数；0 表示所有待处理数据。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset_adapter", default=None)
    parser.add_argument("--model", choices=["double_sphere", "equidistant", "fisheye"], default=None)
    parser.add_argument("--balance", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    args.dry_run = None
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url = database_url_or_error(parser)
    try:
        options = resolved_options_from_yaml_and_args(load_yaml(Path(args.config).expanduser()), args)
        _validate(args.shared_root, options, args.limit)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    from roboego_hand_vis.database import DataService

    def interrupted(_signum, _frame):
        raise SystemExit(143)

    service = DataService.from_url(url)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_pending(service, args.shared_root, options, limit=args.limit)
        print(json.dumps({"stage": "stereo_rectification", "records": results}, indent=2, ensure_ascii=False))
        if any(row["status"] == "failed" for row in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        service.database.close()


if __name__ == "__main__":
    main()
