#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.processing_plan import ProcessingPlanConfig, write_processing_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic processing/drop plan from Stage 1 annotations.")
    parser.add_argument("--stage1_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--actions_json", default="", help="Optional Stage1 actions JSON or directory containing chunk JSON files.")
    parser.add_argument("--min_hold_interruption_s", type=int, default=8)
    parser.add_argument("--min_walk_interruption_s", type=int, default=5)
    parser.add_argument("--min_idle_interruption_s", type=int, default=8)
    parser.add_argument("--context_expand_s", type=int, default=3)
    parser.add_argument("--min_process_segment_s", type=int, default=30)
    parser.add_argument("--merge_process_gap_s", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProcessingPlanConfig(
        min_hold_interruption_s=args.min_hold_interruption_s,
        min_walk_interruption_s=args.min_walk_interruption_s,
        min_idle_interruption_s=args.min_idle_interruption_s,
        context_expand_s=args.context_expand_s,
        min_process_segment_s=args.min_process_segment_s,
        merge_process_gap_s=args.merge_process_gap_s,
    )
    plan = write_processing_plan(
        stage1_json=Path(args.stage1_json),
        actions_json=Path(args.actions_json) if args.actions_json else None,
        output_json=Path(args.output_json),
        config=config,
    )
    summary = {
        "output_json": args.output_json,
        "frame_count": plan["frame_count"],
        "semantic_unit_count": len(plan["semantic_units"]),
        "process_segment_count": len(plan["segments"]),
        "drop_segment_count": len(plan["drop_segments"]),
        "drop_seconds": sum(segment["duration"] for segment in plan["drop_segments"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
