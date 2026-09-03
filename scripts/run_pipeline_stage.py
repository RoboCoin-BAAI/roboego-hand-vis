#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.pipeline_launcher import (
    LauncherRequest,
    StageInvocation,
    WorkflowBlockedError,
    run_launcher_request,
    workflow_stage_names,
)
from roboego_hand_vis.pipeline_state import update_stage_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RoboEgo Kedro stage from a dataset config.")
    parser.add_argument("--dataset", required=True, help="Dataset config name, for example raw.")
    parser.add_argument("--config", default="", help="Optional explicit dataset yaml path.")
    parser.add_argument("--session", required=True, help="Raw session root directory.")
    parser.add_argument("--stage", required=True, help="Stage name from conf/datasets/<dataset>.yml.")
    parser.add_argument("--hand-window", default="", help="Hand-window id or absolute path for hand-window scoped stages.")
    parser.add_argument("--unit", default="", help="Explicit unit root for unit-scoped stages.")
    parser.add_argument("--all-hand-windows", action="store_true", help="Run one command per discovered hand window.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated command(s) without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = " ".join(sys.argv)
    request = LauncherRequest(
        dataset=args.dataset,
        config=Path(args.config).expanduser() if args.config else None,
        session=Path(args.session).expanduser(),
        stage=args.stage,
        unit=args.unit or None,
        hand_window=args.hand_window or None,
        all_hand_windows=bool(args.all_hand_windows),
        dry_run=bool(args.dry_run),
    )
    started = time.monotonic()
    workflow_steps = workflow_stage_names(request)
    is_workflow = workflow_steps != [request.stage]
    step_starts: dict[str, float] = {}

    def record_workflow_stage(
        stage: str,
        status: str,
        invocations: list[StageInvocation],
        error: BaseException | None,
    ) -> None:
        if status == "processing":
            step_starts[stage] = time.monotonic()
            update_stage_state(request.session, stage, "processing", command=command)
            return
        elapsed = time.monotonic() - step_starts.get(stage, started)
        state_status = "blocked" if status == "blocked" else "fail" if status == "fail" else "success"
        update_stage_state(
            request.session,
            stage,
            state_status,
            command=command,
            duration_s=elapsed,
            outputs=[invocation.node for invocation in invocations] if state_status == "success" else None,
            error=str(error) if error is not None else None,
            returncode=2 if state_status == "blocked" else 1 if state_status == "fail" else 0,
        )

    try:
        if not request.dry_run and not is_workflow:
            update_stage_state(request.session, request.stage, "processing", command=command)
        commands = run_launcher_request(request, stage_callback=record_workflow_stage)
    except WorkflowBlockedError as exc:
        if not request.dry_run and not is_workflow:
            update_stage_state(
                request.session,
                request.stage,
                "blocked",
                command=command,
                duration_s=time.monotonic() - started,
                error=str(exc),
                returncode=2,
            )
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    except subprocess.CalledProcessError as exc:
        if not request.dry_run and not is_workflow:
            update_stage_state(
                request.session,
                request.stage,
                "fail",
                command=command,
                duration_s=time.monotonic() - started,
                error=str(exc),
                returncode=exc.returncode,
        )
        raise
    except Exception as exc:
        if not request.dry_run and not is_workflow:
            update_stage_state(
                request.session,
                request.stage,
                "fail",
                command=command,
                duration_s=time.monotonic() - started,
                error=str(exc),
                returncode=1,
        )
        raise
    if not request.dry_run and not is_workflow:
        update_stage_state(
            request.session,
            request.stage,
            "success",
            command=command,
            duration_s=time.monotonic() - started,
            outputs=commands,
            returncode=0,
        )


if __name__ == "__main__":
    main()
