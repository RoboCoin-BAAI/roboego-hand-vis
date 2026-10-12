"""L1 database queue wrapper; the existing session CLI stays independent."""

from __future__ import annotations

import argparse
import json
import signal
from dataclasses import replace
from pathlib import Path, PureWindowsPath

from .cli import DEFAULT_CONFIG, config_from_yaml, load_yaml, run_raw_headband_integrity


def run_pending(service, config, *, limit=0, runner=None) -> list[dict]:
    """Run the existing quality check with no database transaction held open."""
    if not config.gate_enabled:
        raise ValueError("Database mode requires gate.enabled=true")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    root = config.session.resolve()
    if not root.is_dir():
        raise ValueError(f"Shared root is not a directory: {root}")
    runner = runner or run_raw_headband_integrity
    results = []
    while not limit or len(results) < limit:
        row = service.claim_raw_headband_integrity()
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
            summary = runner(replace(config, session=session, output_root=session))
            passed = summary.get("gate", {}).get("passed")
            if passed is not True and passed is not False:
                raise ValueError("Quality check did not return a boolean gate.passed")
            result["summary"] = summary
            result["status"] = "completed" if passed else "block"
        except Exception as exc:
            passed = None
            result["error"] = str(exc)
        except BaseException:
            if not service.finish_raw_headband_integrity(row["id"], passed=None):
                raise RuntimeError(f"Failed to write interrupted task status: {row['id']}")
            raise
        # A writeback failure must stop processing, rather than claim another row.
        if not service.finish_raw_headband_integrity(row["id"], passed=passed):
            raise RuntimeError(f"Failed to write quality status: {row['id']}")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="从 L1 数据库领取已下载的数据，执行头环基础质检。")
    parser.add_argument("--shared-root", required=True, help="下载时使用的共享目录根路径。")
    parser.add_argument("--limit", type=int, default=0, help="最多处理条数；0 表示处理所有待质检数据。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url = database_url_or_error(parser)
    args.session, args.output_root, args.camera_stream = args.shared_root, "", None
    config = config_from_yaml(load_yaml(Path(args.config).expanduser()), args)
    if args.limit < 0 or not config.gate_enabled or not config.session.is_dir():
        parser.error("需要有效的共享目录、非负 limit 和 gate.enabled=true")
    from roboego_hand_vis.database import DataService

    def interrupted(_signum, _frame):
        raise SystemExit(143)

    service = DataService.from_url(url)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_pending(service, config, limit=args.limit)
        print(json.dumps({"stage": "raw_headband_integrity", "records": results},
                         indent=2, ensure_ascii=False))
        if any(row["status"] != "completed" for row in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        service.database.close()


if __name__ == "__main__":
    main()
