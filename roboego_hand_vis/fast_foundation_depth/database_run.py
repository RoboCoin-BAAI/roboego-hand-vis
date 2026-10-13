"""L1 depth-estimation queue; launch the original GPU command unchanged."""

import argparse
import copy
import json
import signal
import sys
from pathlib import Path

import cv2
import yaml

from roboego_hand_vis.common.depth_io import depth_path_for_frame, depth_uint16_png_path_for_frame
from roboego_hand_vis.common.processes import run_logged_command
from roboego_hand_vis.common.stereo_manifest import load_foundation_inputs_from_manifest
from roboego_hand_vis.database.validators import resolve_session
from .cli import DEFAULT_CONFIG, load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _validate(shared_root, data, limit):
    root = Path(shared_root).expanduser().resolve()
    if limit < 0 or not root.is_dir():
        raise ValueError("Need a valid shared root and nonnegative limit")
    data = copy.deepcopy(data)
    paths = data.setdefault("paths", {})
    paths["fast_foundation_root"] = str(Path(paths.get("fast_foundation_root", "../Fast-FoundationStereo")).expanduser().resolve())
    paths["ckpt"] = str(Path(paths.get("ckpt", "../Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth")).expanduser().resolve())
    if not Path(paths["fast_foundation_root"]).is_dir():
        raise ValueError("Fast-FoundationStereo root does not exist")
    if not Path(paths["ckpt"]).is_file() or not Path(paths["ckpt"]).with_name("cfg.yaml").is_file():
        raise ValueError("Model checkpoint or its cfg.yaml does not exist")
    params = data.get("parameters", {})
    start = int(params.get("start", 0))
    count = params.get("num_frames", "auto")
    if start < 0 or (str(count).lower() != "auto" and int(count) <= 0):
        raise ValueError("Invalid start/num_frames")
    if not params.get("save_depth_png", True) and not params.get("save_npy", False):
        raise ValueError("Enable save_depth_png or save_npy")
    return root, data


def _task(session, data):
    manifest = (session / "dataset_manifest/dataset_manifest.json").resolve()
    output = (session / "depth").resolve()
    if not manifest.is_relative_to(session) or not output.is_relative_to(session):
        raise ValueError("Depth input/output must remain inside their session")
    inputs = load_foundation_inputs_from_manifest(manifest)
    counts = []
    for video in (inputs.left_video, inputs.right_video):
        if not video.is_relative_to(session) or not video.is_file():
            raise ValueError("Stereo videos must exist inside their session")
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open stereo video: {video}")
            counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            cap.release()
    params = data.get("parameters", {})
    start = int(params.get("start", 0))
    requested = params.get("num_frames", "auto")
    count = counts[0] - start if str(requested).lower() == "auto" else int(requested)
    if count <= 0 or start + count > min(counts):
        raise ValueError("Requested depth frame range exceeds stereo videos")
    return dict(session=session, manifest=manifest, output_dir=output, frame_indices=range(start, start + count))


def _run_session(task, data):
    annotation = (task["session"] / "annotation").resolve()
    if not annotation.is_relative_to(task["session"]):
        raise ValueError("Annotation output must remain inside its session")
    annotation.mkdir(parents=True, exist_ok=True)
    config_path = annotation / "fast_foundation_depth.database.yaml"
    if not config_path.resolve().is_relative_to(task["session"]):
        raise ValueError("Runtime configuration must remain inside its session")
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))
    log = annotation / "fast_foundation_depth.log"
    if not log.resolve().is_relative_to(task["session"]):
        raise ValueError("Log must remain inside its session")
    command = [sys.executable, "-m", "roboego_hand_vis.fast_foundation_depth.run", "--config", str(config_path),
               "--manifest", str(task["manifest"]), "--output_dir", str(task["output_dir"])]
    run_logged_command(command, log, cwd=PROJECT_ROOT)
    return {"status": "complete", "log_path": str(log)}


def _check_outputs(task, data):
    params = data.get("parameters", {})
    meta = json.loads((task["output_dir"] / "fast_foundation_stereo_video_meta.json").read_text())
    requested = params.get("num_frames", "auto")
    count = None if str(requested).lower() == "auto" else int(requested)
    if (meta.get("manifest_path") != str(task["manifest"]) or meta.get("start") != int(params.get("start", 0))
            or meta.get("num_frames") != count):
        raise RuntimeError("Depth metadata does not match the current task")
    for idx in task["frame_indices"]:
        outputs = []
        if params.get("save_depth_png", True):
            outputs.append(depth_uint16_png_path_for_frame(task["output_dir"], idx))
        if params.get("save_npy", False):
            outputs.append(depth_path_for_frame(task["output_dir"], idx))
        for path in outputs:
            if not path.resolve().is_relative_to(task["session"]) or not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or invalid depth frame: {idx}")


def run_pending(service, shared_root, data, *, limit=0, runner=None):
    root, data = _validate(shared_root, data, limit)
    runner = runner or _run_session
    results = []
    while not limit or len(results) < limit:
        row = service.claim_fast_foundation_depth()
        if row is None:
            break
        result = {"id": row["id"], "status": "failed"}
        succeeded = False
        try:
            session = resolve_session(root, row["local_path"])
            task = _task(session, data)
            summary = runner(task, data)
            if summary.get("status") != "complete":
                raise RuntimeError("Depth estimation did not report completion")
            _check_outputs(task, data)
            result.update(status="completed", frame_count=len(task["frame_indices"]), summary=summary)
            succeeded = True
        except Exception as exc:
            result["error"] = str(exc)
        except BaseException:
            if not service.finish_fast_foundation_depth(row["id"], succeeded=False):
                raise RuntimeError(f"Failed to write interrupted depth status: {row['id']}")
            raise
        if not service.finish_fast_foundation_depth(row["id"], succeeded=succeeded):
            raise RuntimeError(f"Failed to write depth status: {row['id']}")
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="从所选数据表领取深度前门禁已通过的数据，运行 Fast-FoundationStereo。")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    from roboego_hand_vis.database.cli import add_table_argument, stage_service_or_error
    add_table_argument(parser)
    args = parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url = database_url_or_error(parser)
    try:
        data = load_yaml(Path(args.config).expanduser())
        _validate(args.shared_root, data, args.limit)
        import torch
        if not torch.cuda.is_available():
            parser.error("Fast-FoundationStereo requires CUDA; check CUDA_VISIBLE_DEVICES and your environment")
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    service = stage_service_or_error(parser, url, args.table, 'fast_foundation_depth', args=args)
    def interrupted(_signum, _frame):
        raise SystemExit(143)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_pending(service, args.shared_root, data, limit=args.limit)
        print(json.dumps({"stage": "fast_foundation_depth", "records": results}, indent=2, ensure_ascii=False))
        if any(r["status"] != "completed" for r in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if hasattr(service, "workspace"):
            service.workspace.release()
        service.database.close()


if __name__ == "__main__":
    main()
