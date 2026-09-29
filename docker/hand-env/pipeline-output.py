"""Compact delivery outputs for the unchanged documented hand pipeline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def validate_artifacts(final, video):
    import cv2
    import pyarrow.parquet as pq

    def frames(path):
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot read video: {path}")
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

    expected = frames(video)
    count = pq.read_metadata(final / "hand_tracking.parquet").num_rows
    rendered = frames(final / "render.mp4")
    if expected <= 0 or count != expected or rendered != expected:
        raise ValueError(f"Frame mismatch: input={expected}, parquet={count}, render={rendered}")
    summary = final / "summary.json"
    return {"frame_count": count, "render_frame_count": rendered,
            "summary": json.loads(summary.read_text()) if summary.exists() else {}}


def revision(path):
    prefix = ["git", "-c", f"safe.directory={path}", "-C", path]
    result = subprocess.run([*prefix, "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode:
        return {"commit": None}
    dirty = subprocess.run([*prefix, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    return {"commit": result.stdout.strip(), "tracked_changes": bool(dirty.stdout.strip())}


def execute(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("depth", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("max_frames", nargs="?", type=int, default=10000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    work = output / ".work"
    work.mkdir()
    report = {"schema_version": "roboego.hand.delivery.v1", "status": "failed",
              "quality_status": "not_validated", "debug": args.debug,
              "input": {"video": str(args.video), "depth": str(args.depth), "max_frames": args.max_frames},
              "image_id": os.environ.get("HAND_IMAGE_ID"), "artifacts": {}}
    started = time.monotonic()
    code = 1
    try:
        report["sources"] = {name: revision(path) for name, path in {
            "mint": "/home/user/wuji-ego-mint", "egohandkit": "/home/user/EgoHandKit",
            "project": "/home/user/roboego-hand-vis"}.items()}
        command = ["bash", str(Path(__file__).with_name("pipeline-stages.sh")),
                   str(args.video.resolve()), str(args.depth.resolve()), str(work / "run"), str(args.max_frames)]
        print("Running MINT + EgoHandKit; final results will be published when complete.", flush=True)
        with (work / "console.log").open("w") as log:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
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
        if code:
            raise RuntimeError(f"Hand pipeline exited with code {code}; see error.log")
        candidates = list((work / "run/mint").glob("*/final/hand_tracking.parquet"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one final hand tracking file, found {len(candidates)}")
        final = candidates[0].parent
        report.update(validate_artifacts(final, args.video))
        for name in ("hand_tracking.parquet", "render.mp4"):
            shutil.copyfile(final / name, output / name)
        report["artifacts"] = {"hand_tracking": "hand_tracking.parquet", "render": "render.mp4"}
        report["status"] = "complete"
    except BaseException as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else (code or 1)
        report["error"] = str(error) or type(error).__name__
        for name in ("hand_tracking.parquet", "render.mp4"):
            (output / name).unlink(missing_ok=True)
    finally:
        report["returncode"] = code
        report["wall_duration_s"] = time.monotonic() - started
        if code:
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
    print(f"Hand pipeline {report['status']}: {output}/report.json", flush=True)
    return code


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(execute())
