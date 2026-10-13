"""L1 annotation queue; reuse the original Qwen pipeline and checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path, PureWindowsPath

from .cli import DEFAULT_CONFIG, config_from_yaml, load_yaml
from .compute import run_qwen_annotation


def _session_config(data, session):
    return config_from_yaml(data, video=session / "videos" / "left.mkv",
                            output_json=session / "annotation" / "qwen_frame_annotations_left.json")


def _validate(shared_root, data, limit):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    root = Path(shared_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Shared root is not a directory: {root}")
    # The original single-video CLI permits fixed output directories. Reusing
    # those for a queue would mix frames/checkpoints across different sessions.
    parameters = data.get("parameters", {})
    for field in ("frames_dir", "raw_output_dir", "checkpoint_json", "batch_output_dir",
                  "chunk_output_dir", "windows_json"):
        if parameters.get(field):
            raise ValueError(f"Database mode requires {field} to be unset; paths are assigned per session")
    config = _session_config(data, root)
    if not config.prompt.is_file():
        raise ValueError(f"Qwen prompt not found: {config.prompt}")
    if not os.environ.get(config.api_key_env):
        raise ValueError(f"{config.api_key_env} is not set")
    return root


def _check_complete(summary, config):
    if (summary.get("status") != "complete" or summary.get("failed_batch_count") != 0
            or not isinstance(summary.get("frame_count"), int) or summary["frame_count"] <= 0):
        raise RuntimeError("Qwen annotation is incomplete or contains failed batches")
    payload = json.loads(config.output_json.read_text(encoding="utf-8"))
    if (payload.get("status") != "complete" or payload.get("failed_batch_count") != 0
            or payload.get("failed_batches") or not isinstance(payload.get("frames"), list)
            or len(payload["frames"]) != summary["frame_count"]
            or payload.get("source_video") != str(config.video)):
        raise RuntimeError("Qwen output is incomplete or does not match the current video")


def run_pending(service, shared_root, data, *, limit=0, runner=None) -> list[dict]:
    """Partial API/parse failures fail the task; keep files for explicit resume."""
    root = _validate(shared_root, data, limit)
    runner = runner or run_qwen_annotation
    results = []
    while not limit or len(results) < limit:
        row = service.claim_qwen_annotation()
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
            config = _session_config(data, session)
            if (not session.is_relative_to(root) or not session.is_dir()
                    or not config.video.is_relative_to(root) or not config.video.is_file()):
                raise ValueError(f"Session/left video is missing or outside the shared root: {local}")
            for output in (config.output_json, config.frames_dir, config.raw_output_dir,
                           config.batch_output_dir, config.chunk_output_dir, config.checkpoint_json):
                if not output.resolve().is_relative_to(session):
                    raise ValueError("Qwen output paths must remain inside their session")
            summary = runner(config)
            result["summary"] = summary
            _check_complete(summary, config)
            result["status"] = "completed"
            succeeded = True
        except Exception as exc:
            result["error"] = str(exc)
        except BaseException:
            if not service.finish_qwen_annotation(row["id"], succeeded=False):
                raise RuntimeError(f"Failed to write interrupted task status: {row['id']}")
            raise
        if not service.finish_qwen_annotation(row["id"], succeeded=succeeded):
            raise RuntimeError(f"Failed to write annotation status: {row['id']}")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="从所选数据表领取视频质检已通过的数据，标注原始左目视频。")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--limit", type=int, default=0, help="最多处理条数；0 表示所有待标注数据。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    from roboego_hand_vis.database.cli import add_table_argument, stage_service_or_error
    add_table_argument(parser)
    args = parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url = database_url_or_error(parser)
    try:
        data = load_yaml(Path(args.config).expanduser())
        _validate(args.shared_root, data, args.limit)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    def interrupted(_signum, _frame):
        raise SystemExit(143)

    service = stage_service_or_error(parser, url, args.table, 'qwen_annotation', args=args)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_pending(service, args.shared_root, data, limit=args.limit)
        print(json.dumps({"stage": "qwen_annotation", "records": results}, indent=2, ensure_ascii=False))
        if any(row["status"] == "failed" for row in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if hasattr(service, "workspace"):
            service.workspace.release()
        service.database.close()


if __name__ == "__main__":
    main()
