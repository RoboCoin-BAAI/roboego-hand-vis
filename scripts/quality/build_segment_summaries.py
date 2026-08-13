#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.processing_plan import write_segment_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build segment-level summaries from processing plan and Stage 1 annotations.")
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--stage1_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--actions_json", default="")
    parser.add_argument("--selected_segments_dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_segment_summaries(
        plan_json=Path(args.plan_json),
        stage1_json=Path(args.stage1_json),
        actions_json=Path(args.actions_json) if args.actions_json else None,
        output_json=Path(args.output_json),
        selected_segments_dir=Path(args.selected_segments_dir) if args.selected_segments_dir else None,
    )
    summary = {
        "output_json": args.output_json,
        "segment_count": payload["segment_count"],
        "per_segment_dir": args.selected_segments_dir or None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
