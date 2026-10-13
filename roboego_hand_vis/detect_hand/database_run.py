"""Optional L1 hand detection queue, independent of quality gate decisions."""

from __future__ import annotations

import argparse
import json
import os
import signal
from dataclasses import replace
from pathlib import Path, PureWindowsPath

from .cli import (DEFAULT_CONFIG, config_from_yaml, default_output_json_for_video,
                  load_yaml, run_detect_hand_stage)


def _validate(shared_root, config, limit):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    root = Path(shared_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Shared root is not a directory: {root}")
    if not config.model.is_file():
        raise ValueError(f"Hand detector model not found: {config.model}")
    return root


def run_pending(service, shared_root, config, *, limit=0, runner=None) -> list[dict]:
    """A successful detection completes regardless of hand presence ratios."""
    root = _validate(shared_root, config, limit)
    runner = runner or run_detect_hand_stage
    results = []
    while not limit or len(results) < limit:
        row = service.claim_detect_hand()
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
            video = (session / "videos" / "left.mkv").resolve()
            if (not session.is_relative_to(root) or not session.is_dir()
                    or not video.is_relative_to(root) or not video.is_file()):
                raise ValueError(f"Session/left video is missing or outside the shared root: {local}")
            summary = runner(replace(config, video=video,
                                     output_json=default_output_json_for_video(session / "videos" / "left.mkv")))
            result["summary"] = summary
            result["status"] = "completed"
            succeeded = True
        except Exception as exc:
            result["error"] = str(exc)
        except BaseException:
            if not service.finish_detect_hand(row["id"], succeeded=False):
                raise RuntimeError(f"Failed to write interrupted task status: {row['id']}")
            raise
        if not service.finish_detect_hand(row["id"], succeeded=succeeded):
            raise RuntimeError(f"Failed to write detection status: {row['id']}")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="从所选数据表领取视频质检已通过的数据，检测左目手部出现比例。")
    parser.add_argument("--shared-root", required=True, help="下载时使用的共享目录根路径。")
    parser.add_argument("--limit", type=int, default=0, help="最多处理条数；0 表示所有待检测数据。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--sample_every", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    from roboego_hand_vis.database.cli import add_table_argument, stage_service_or_error
    add_table_argument(parser)
    args = parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url = database_url_or_error(parser)
    args.video, args.output_json = str(Path(args.shared_root) / "videos" / "left.mkv"), ""
    try:
        config = config_from_yaml(load_yaml(Path(args.config).expanduser()), args)
        _validate(args.shared_root, config, args.limit)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    def interrupted(_signum, _frame):
        raise SystemExit(143)

    service = stage_service_or_error(parser, url, args.table, 'detect_hand', args=args)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_pending(service, args.shared_root, config, limit=args.limit)
        print(json.dumps({"stage": "detect_hand", "records": results}, indent=2, ensure_ascii=False))
        if any(row["status"] == "failed" for row in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if hasattr(service, "workspace"):
            service.workspace.release()
        service.database.close()


if __name__ == "__main__":
    main()
