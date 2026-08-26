#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.pipeline_launcher import LauncherRequest, build_stage_invocations, run_stage_invocations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RoboEgo Kedro stage from a dataset config.")
    parser.add_argument("--dataset", required=True, help="Dataset config name, for example wumei.")
    parser.add_argument("--config", default="", help="Optional explicit dataset yaml path.")
    parser.add_argument("--session", required=True, help="Raw session root directory.")
    parser.add_argument("--stage", required=True, help="Stage name from conf/datasets/<dataset>.yml.")
    parser.add_argument("--hand-window", default="", help="Hand-window id or absolute path for hand-window scoped stages.")
    parser.add_argument("--all-hand-windows", action="store_true", help="Run one command per discovered hand window.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated command(s) without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = LauncherRequest(
        dataset=args.dataset,
        config=Path(args.config).expanduser() if args.config else None,
        session=Path(args.session).expanduser(),
        stage=args.stage,
        hand_window=args.hand_window or None,
        all_hand_windows=bool(args.all_hand_windows),
        dry_run=bool(args.dry_run),
    )
    invocations = build_stage_invocations(request)
    run_stage_invocations(invocations, dry_run=request.dry_run)


if __name__ == "__main__":
    main()
