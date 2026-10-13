"""Database queue entry for the existing session video quality gate."""

from __future__ import annotations

import argparse
import json
import signal
from dataclasses import replace
from pathlib import Path, PureWindowsPath

from .cli import (DEFAULT_CONFIG, config_from_yaml, default_output_json_for_session,
                  load_yaml, run_session_video_simple_quality)


def _shared_root(config, limit):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if len(config.streams) != 2 or set(config.streams) != {"left", "right"}:
        raise ValueError("Database gate must check both left and right streams")
    if config.session is None:
        raise ValueError("Shared root must be configured")
    root = config.session.resolve()
    if not root.is_dir():
        raise ValueError(f"Shared root is not a directory: {root}")
    return root


def run_pending(service, config, *, limit=0, runner=None) -> list[dict]:
    """Run the session gate without keeping a database transaction open."""
    root = _shared_root(config, limit)
    runner = runner or run_session_video_simple_quality
    results = []
    while not limit or len(results) < limit:
        row = service.claim_video_simple_quality()
        if row is None:
            break
        result = {"id": row["id"], "status": "failed"}
        passed = None
        try:
            local = row["local_path"]
            path = Path(local)
            if (not local.strip() or path.is_absolute() or ".." in path.parts
                    or "\\" in local or PureWindowsPath(local).drive or str(path) == "."):
                raise ValueError("local_path must be relative to the shared root")
            session = (root / path).resolve()
            if not session.is_relative_to(root) or not session.is_dir():
                raise ValueError(f"Session is missing or outside the shared root: {local}")
            summary = runner(replace(config, session=session, video=None,
                                     output_json=default_output_json_for_session(session)))
            passed = summary.get("passed")
            if passed is not True and passed is not False:
                raise ValueError("Quality check did not return a boolean passed")
            result["summary"] = summary
            result["status"] = "completed" if passed else "block"
        except Exception as exc:
            passed = None
            result["error"] = str(exc)
        except BaseException:
            if not service.finish_video_simple_quality(row["id"], passed=None):
                raise RuntimeError(f"Failed to write interrupted task status: {row['id']}")
            raise
        if not service.finish_video_simple_quality(row["id"], passed=passed):
            raise RuntimeError(f"Failed to write quality status: {row['id']}")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="从所选数据表领取基础质检已通过的数据，执行视频粗质检。")
    parser.add_argument("--shared-root", required=True, help="下载时使用的共享目录根路径。")
    parser.add_argument("--limit", type=int, default=0, help="最多处理条数；0 表示所有待质检数据。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    from roboego_hand_vis.database.cli import add_table_argument, stage_service_or_error
    add_table_argument(parser)
    args = parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url = database_url_or_error(parser)
    args.session, args.video, args.output_json = args.shared_root, "", ""
    try:
        config = config_from_yaml(load_yaml(Path(args.config).expanduser()), args)
        _shared_root(config, args.limit)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    def interrupted(_signum, _frame):
        raise SystemExit(143)

    service = stage_service_or_error(parser, url, args.table, 'video_simple_quality', args=args)
    config = replace(config, session=service.work_root)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_pending(service, config, limit=args.limit)
        print(json.dumps({"stage": "video_simple_quality", "records": results},
                         indent=2, ensure_ascii=False))
        if any(row["status"] != "completed" for row in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if hasattr(service, "workspace"):
            service.workspace.release()
        service.database.close()


if __name__ == "__main__":
    main()
