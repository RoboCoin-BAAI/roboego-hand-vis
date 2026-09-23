#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.hand_windows.logic import HandWindowConfig, write_hand_estimation_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict hand-estimation windows from Stage1 actions and processing plan.")
    parser.add_argument("--stage1_actions", required=True)
    parser.add_argument("--processing_plan", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--context_expand_s", type=int, default=1)
    parser.add_argument("--merge_gap_s", type=int, default=2)
    parser.add_argument("--merge_after_expand", action="store_true")
    parser.add_argument("--min_window_s", type=int, default=1)
    parser.add_argument("--min_priority", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--min_core_duration_s", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--require_both_hands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hand_window_mode",
        choices=["two", "single", "single_and_two", "any"],
        default="two",
        help=(
            "two: only left+right visible windows; single: only one-hand windows; "
            "single_and_two: split single-hand and two-hand windows with recommended policies; "
            "any: any visible hand window."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_hand_estimation_windows(
        stage1_actions_json=Path(args.stage1_actions).expanduser(),
        processing_plan_json=Path(args.processing_plan).expanduser(),
        output_json=Path(args.output_json).expanduser(),
        config=HandWindowConfig(
            require_both_hands=args.require_both_hands,
            hand_window_mode=args.hand_window_mode,
            context_expand_s=args.context_expand_s,
            merge_gap_s=args.merge_gap_s,
            merge_after_expand=args.merge_after_expand,
            min_window_s=args.min_window_s,
            min_priority=args.min_priority,
            min_core_duration_s=args.min_core_duration_s,
            fps=args.fps,
        ),
    )
    counts = Counter(str(window.get("window_type", "unknown")) for window in payload.get("windows", []))
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "window_count": payload["window_count"],
                "window_type_counts": dict(sorted(counts.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
