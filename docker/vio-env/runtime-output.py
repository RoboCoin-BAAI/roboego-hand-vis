"""Publish a small output contract around the unchanged project VIO CLI."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np


def execute(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("/data/output"))
    parser.add_argument("--session", default=os.environ.get("INPUT_CONTAINER_PATH", "/data/input"))
    parser.add_argument("--time-offset-mode", choices=("estimate", "config"), default="estimate")
    parser.add_argument("--timing-bootstrap", choices=("auto", "off"), default="auto")
    args, extra = parser.parse_known_args(argv)
    if "--prepare_only" in extra:
        parser.error("Use the prepare action separately; run publishes a trajectory")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    # Atomic directory creation also excludes simultaneous writers.
    work = output / ".work"
    work.mkdir()
    command = [sys.executable, "-m", "roboego_hand_vis.vio.run", *extra,
               "--session", args.session, "--output_dir", str(work / "run"),
               "--openvins_root", "/opt/open_vins",
               "--openvins_binary", "/opt/open_vins/ov_msckf/build/roboego_openvins_offline"]
    report = {"schema_version": "roboego.vio.delivery.v2", "status": "failed",
              "quality_status": "not_validated", "pose_count": 0,
              "input": {"session": args.session, "arguments": extra},
              "debug": args.debug, "image_id": os.environ.get("VIO_IMAGE_ID"),
              "time_offset_mode": args.time_offset_mode,
              "timing_bootstrap_mode": args.timing_bootstrap,
              "project_revision": os.environ.get("VIO_PROJECT_REVISION"),
              "openvins_revision": os.environ.get("VIO_OPENVINS_REVISION"), "artifacts": {}}
    started = time.monotonic()
    code = 1
    try:
        with (work / "console.log").open("w") as log:
            environment = dict(os.environ, OPENVINS_TIME_OFFSET_MODE=args.time_offset_mode,
                               OPENVINS_TIMING_BOOTSTRAP=args.timing_bootstrap)
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, env=environment) as process:
                try:
                    code = process.wait()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
        native_report = work / "run/trajectory/run_report.json"
        if native_report.exists():
            native = json.loads(native_report.read_text())
            for key in ("pose_count", "metrics", "offline", "covariance_exported", "frame_alignment"):
                if key in native:
                    report[key] = native[key]
            report["execution_status"] = native.get("status", "unknown")
        trajectory = work / "run/trajectory/trajectory.npz"
        if trajectory.is_file() and report.get('frame_alignment'):
            alignment = report['frame_alignment']
            with np.load(trajectory, allow_pickle=False) as data:
                count = alignment['frame_count']
                valid = data['status'] == 'valid'
                if (str(data['schema_version']) != 'openvins.frame_aligned.v2'
                        or not np.array_equal(data['frame_index'], np.arange(count))
                        or {'valid', 'processed', 'initialized'}.intersection(data.files)
                        or data['status'].dtype.kind != 'U'
                        or data['status'].shape != (count,)
                        or int(valid.sum()) != report['pose_count']
                        or alignment['valid_pose_count'] != report['pose_count']):
                    raise RuntimeError('Invalid frame-aligned trajectory contract')
                for key in ('T_world_imu', 'T_world_camera'):
                    if (data[key].shape != (count, 4, 4)
                            or not np.isfinite(data[key][valid]).all()
                            or not np.isnan(data[key][~valid]).all()):
                        raise RuntimeError('Invalid pose or missing NaN for unavailable frame')
            # A failed run can still deliver useful per-frame failure states.
            shutil.copyfile(trajectory, output / 'trajectory.npz')
            report['artifacts']['trajectory'] = 'trajectory.npz'
        if code == 0:
            if 'trajectory' not in report['artifacts'] or report["pose_count"] < 2:
                raise RuntimeError("Runner exited successfully without a trajectory artifact")
            report["status"] = "complete"
        else:
            report["error"] = f"VIO process exited with code {code}; see error.log"
    except BaseException as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else (code or 1)
        report["status"] = "failed"
        report["error"] = str(error) or type(error).__name__
    finally:
        report["returncode"] = code
        report["wall_duration_s"] = time.monotonic() - started
        if code != 0:
            log = work / "console.log"
            if log.exists():
                shutil.copyfile(log, output / "error.log")
            else:
                (output / "error.log").write_text(report.get("error", "Run failed") + "\n")
            report["artifacts"]["error_log"] = "error.log"
        if args.debug:
            work.rename(output / "debug")
            report["artifacts"]["debug"] = "debug/"
        else:
            shutil.rmtree(work)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"VIO {report['status']}: {output}/report.json", flush=True)
    return code


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(execute())
