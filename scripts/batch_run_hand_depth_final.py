#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.pipeline_state import discover_sessions, read_pipeline_state


REQUIRED_SUCCESS_STAGES = ("hamer", "fast_foundation_depth")
FINAL_HAND_DEPTH_STAGES = (
    "hand_filter",
    "joints21_pose_gate",
    "depth_anchored_joints21",
    "final_hand_result",
    "final_hand_trajectory_polish",
)


def build_stage_command(
    *,
    python_bin: str,
    dataset: str,
    session: Path,
    stage: str,
    config: Path | None,
    force: bool,
) -> list[str]:
    cmd = [
        python_bin,
        "scripts/run_pipeline_stage.py",
        "--dataset",
        dataset,
        "--session",
        str(Path(session).expanduser().resolve()),
        "--stage",
        stage,
    ]
    if config is not None:
        cmd.extend(["--config", str(config)])
    if force:
        cmd.append("--force")
    return cmd


def run_one_session(
    *,
    session: Path,
    dataset: str,
    python_bin: str,
    config: Path | None,
    force: bool,
    dry_run: bool,
    exit_on_error: bool,
) -> dict[str, Any]:
    session = Path(session).expanduser().resolve()
    state = read_pipeline_state(session)
    stages = state.get("stages", {}) if isinstance(state.get("stages"), dict) else {}
    required_statuses = {
        stage: (stages.get(stage) or {}).get("status") if isinstance(stages.get(stage), dict) else "missing"
        for stage in REQUIRED_SUCCESS_STAGES
    }
    if any(status != "success" for status in required_statuses.values()):
        return {
            "session": str(session),
            "status": "skipped",
            "reason": "required_stage_not_success",
            "required_statuses": required_statuses,
        }

    results: list[dict[str, Any]] = []
    for stage in FINAL_HAND_DEPTH_STAGES:
        state = read_pipeline_state(session)
        stages = state.get("stages", {}) if isinstance(state.get("stages"), dict) else {}
        record = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        if not force and record.get("status") == "success":
            results.append({"stage": stage, "status": "skipped", "reason": "already_success"})
            continue

        cmd = build_stage_command(
            python_bin=python_bin,
            dataset=dataset,
            session=session,
            stage=stage,
            config=config,
            force=force,
        )
        if dry_run:
            print(" ".join(cmd))
            results.append({"stage": stage, "status": "skipped", "reason": "dry_run"})
            continue

        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        status = "success" if completed.returncode == 0 else "blocked" if completed.returncode == 2 else "fail"
        results.append({"stage": stage, "status": status, "returncode": int(completed.returncode)})
        if completed.returncode != 0:
            if exit_on_error:
                raise subprocess.CalledProcessError(completed.returncode, cmd)
            break

    stage_counts = Counter(result["status"] for result in results)
    failed = [result for result in results if result["status"] in {"blocked", "fail"}]
    return {
        "session": str(session),
        "status": "blocked" if failed and failed[0]["status"] == "blocked" else "fail" if failed else "success",
        "stage_counts": dict(stage_counts),
        "stages": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final hand-depth stages for sessions whose HaMeR and depth stages succeeded."
    )
    parser.add_argument("--root", required=True, help="Session root or parent root containing cp_*/session_*/divide/session_*.")
    parser.add_argument("--dataset", default="raw")
    parser.add_argument("--config", default="")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run scripts/run_pipeline_stage.py.")
    parser.add_argument("--force", action="store_true", help="Re-run stages already marked success.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--exit-on-error", action="store_true", help="Stop at the first failed or blocked session.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    config = Path(args.config).expanduser() if args.config else None
    sessions = discover_sessions(root)
    results: list[dict[str, Any]] = []
    for session in sessions:
        result = run_one_session(
            session=session,
            dataset=args.dataset,
            python_bin=args.python,
            config=config,
            force=bool(args.force),
            dry_run=bool(args.dry_run),
            exit_on_error=bool(args.exit_on_error),
        )
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "stages"}, ensure_ascii=False))
        if args.exit_on_error and result["status"] in {"blocked", "fail"}:
            break

    summary = {
        "root": str(root),
        "dataset": args.dataset,
        "session_count": len(sessions),
        "status_counts": dict(Counter(result["status"] for result in results)),
        "stages": list(FINAL_HAND_DEPTH_STAGES),
        "required_success_stages": list(REQUIRED_SUCCESS_STAGES),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if any(result["status"] == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
