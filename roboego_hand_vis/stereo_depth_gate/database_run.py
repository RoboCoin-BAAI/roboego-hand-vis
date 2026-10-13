"""Independent L1 queue entry for pre-depth stereo quality checking."""

import argparse
import json
import signal
from pathlib import Path

from roboego_hand_vis.database.validators import resolve_session
from .cli import DEFAULT_CONFIG, config_from_yaml_and_args, load_yaml, run_stereo_depth_gate


def run_pending(service, shared_root, data, *, limit=0, runner=None):
    root = Path(shared_root).expanduser().resolve()
    if limit < 0 or not root.is_dir():
        raise ValueError("Need a valid shared root and nonnegative limit")
    runner = runner or run_stereo_depth_gate
    results = []
    while not limit or len(results) < limit:
        row = service.claim_stereo_depth_gate()
        if row is None:
            break
        result = {"id": row["id"], "status": "failed"}
        passed = None
        try:
            session = resolve_session(root, row["local_path"])
            args = argparse.Namespace(session=str(session), manifest=str(session / "dataset_manifest/dataset_manifest.json"),
                output_dir=str(session / "quality/stereo_depth_gate"), sample_count=None, min_orb_matches=None)
            config = config_from_yaml_and_args(data, args)
            if not config.manifest_path.is_relative_to(session) or not config.manifest_path.is_file():
                raise ValueError("Manifest is missing or outside its session")
            if not config.output_dir.is_relative_to(session):
                raise ValueError("Gate output must remain inside its session")
            # Check source video paths before the original gate reads any frames.
            manifest = json.loads(config.manifest_path.read_text())
            for eye in ("left", "right"):
                video = Path(manifest["videos"][eye]["path"]).expanduser().resolve()
                if not video.is_relative_to(session) or not video.is_file():
                    raise ValueError("Stereo videos must exist inside their session")
            report = runner(config)
            passed = report.get("usable_for_depth")
            if passed is not True and passed is not False:
                raise ValueError("Gate did not return boolean usable_for_depth")
            result.update(status="completed" if passed else "block", summary=report)
        except Exception as exc:
            passed = None
            result["error"] = str(exc)
        except BaseException:
            if not service.finish_stereo_depth_gate(row["id"], passed=None):
                raise RuntimeError(f"Failed to write interrupted gate status: {row['id']}")
            raise
        if not service.finish_stereo_depth_gate(row["id"], passed=passed):
            raise RuntimeError(f"Failed to write gate status: {row['id']}")
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="从所选数据表领取立体矫正已完成的数据，运行深度前门禁。")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    from roboego_hand_vis.database.cli import add_table_argument, stage_service_or_error
    add_table_argument(parser)
    args = parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url = database_url_or_error(parser)
    data = load_yaml(Path(args.config).expanduser())
    if args.limit < 0 or not Path(args.shared_root).expanduser().is_dir():
        parser.error("Need a valid shared root and nonnegative limit")
    service = stage_service_or_error(parser, url, args.table, 'stereo_depth_gate', args=args)
    def interrupted(_signum, _frame):
        raise SystemExit(143)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = run_pending(service, args.shared_root, data, limit=args.limit)
        print(json.dumps({"stage": "stereo_depth_gate", "records": results}, indent=2, ensure_ascii=False))
        if any(r["status"] != "completed" for r in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if hasattr(service, "workspace"):
            service.workspace.release()
        service.database.close()


if __name__ == "__main__":
    main()
