"""L1 queue wrapper for the existing post-annotation standardization stage."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path, PureWindowsPath

from .cli import DEFAULT_CONFIG, config_from_yaml, load_yaml
from .compute import run_post_annotation_actions


def _session_config(data, session, overwrite):
    annotation = session / "annotation"
    return config_from_yaml(data, qwen_chunk_dir=annotation / "qwen_frame_annotation_chunks",
                            output_dir=annotation, overwrite=overwrite)


def _validate(shared_root, data, limit, overwrite):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    root = Path(shared_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Shared root is not a directory: {root}")
    for field in ("action_chunk_dir", "review_chunk_dir", "resolved_chunk_dir",
                  "merged_output_json", "subtask_output_json", "frames_dir"):
        if data.get("parameters", {}).get(field):
            raise ValueError(f"Database mode requires {field} to be unset; paths are assigned per session")
    config = _session_config(data, root, overwrite)
    if not os.environ.get(config.api_key_env):
        raise ValueError(f"{config.api_key_env} is not set")
    if config.generate_subtasks:
        for prompt in (config.subtask_prompt, config.subtask_object_prompt):
            if not prompt.is_file():
                raise ValueError(f"Subtask prompt not found: {prompt}")
    return root


def _check_complete(summary, config, chunks):
    count = summary.get("merged_frame_count")
    if summary.get("status") != "complete" or not isinstance(count, int) or count <= 0:
        raise RuntimeError("Post-annotation output is incomplete or contains no frames")
    for chunk in chunks:
        payload = json.loads((config.resolved_chunk_dir / chunk.name).read_text(encoding="utf-8"))
        if not isinstance(payload.get("frames"), list):
            raise RuntimeError(f"Invalid resolved chunk: {chunk.name}")
        if payload.get("vl_object_resolver_status") not in (None, "ok"):
            raise RuntimeError(f"VL object resolution failed for {chunk.name}")
    merged = json.loads(config.merged_output_json.read_text(encoding="utf-8"))
    if (not isinstance(merged.get("frames"), list) or len(merged["frames"]) != count
            or merged.get("frame_count") != count):
        raise RuntimeError("Merged actions do not match the completion summary")
    if config.generate_subtasks:
        subtasks = json.loads(config.subtask_output_json.read_text(encoding="utf-8"))
        if not isinstance(subtasks.get("subtasks"), list):
            raise RuntimeError("Subtask output is missing or invalid")


def run_pending(service, shared_root, data, *, limit=0, overwrite=None, runner=None) -> list[dict]:
    root = _validate(shared_root, data, limit, overwrite)
    runner = runner or run_post_annotation_actions
    results = []
    while not limit or len(results) < limit:
        row = service.claim_post_annotation_actions()
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
            config = _session_config(data, session, overwrite)
            for target in (config.qwen_chunk_dir, config.action_chunk_dir, config.review_chunk_dir,
                           config.resolved_chunk_dir, config.merged_output_json,
                           config.subtask_output_json, config.frames_dir):
                if target is not None and not target.resolve().is_relative_to(session):
                    raise ValueError("Post-annotation paths must remain inside their session")
            chunks = sorted(p for p in config.qwen_chunk_dir.glob("chunk_*.json") if p.is_file())
            if not chunks:
                raise RuntimeError("No Qwen annotation chunks found")
            if any(not p.resolve().is_relative_to(session) for p in chunks):
                raise ValueError("Qwen chunks must remain inside their session")
            summary = runner(config)
            result["summary"] = summary
            _check_complete(summary, config, chunks)
            result["status"] = "completed"
            succeeded = True
        except Exception as exc:
            result["error"] = str(exc)
        except BaseException:
            if not service.finish_post_annotation_actions(row["id"], succeeded=False):
                raise RuntimeError(f"Failed to write interrupted task status: {row['id']}")
            raise
        if not service.finish_post_annotation_actions(row["id"], succeeded=succeeded):
            raise RuntimeError(f"Failed to write post-annotation status: {row['id']}")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="从所选数据表领取千问标注已完成的数据，整理标准字段。")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    from roboego_hand_vis.database.cli import add_table_argument, stage_service_or_error
    add_table_argument(parser)
    args = parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url = database_url_or_error(parser)
    try:
        data = load_yaml(Path(args.config).expanduser())
        _validate(args.shared_root, data, args.limit, args.overwrite)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    def interrupted(_signum, _frame):
        raise SystemExit(143)

    service = stage_service_or_error(parser, url, args.table, 'post_annotation_actions', args=args)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_pending(service, args.shared_root, data, limit=args.limit, overwrite=args.overwrite)
        print(json.dumps({"stage": "post_annotation_actions", "records": results}, indent=2, ensure_ascii=False))
        if any(row["status"] == "failed" for row in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if hasattr(service, "workspace"):
            service.workspace.release()
        service.database.close()


if __name__ == "__main__":
    main()
