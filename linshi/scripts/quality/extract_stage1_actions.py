#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.post_annotation_actions.stage1_actions import write_stage1_action_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract atomic action candidates from Stage 1 Qwen annotations.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_stage1_action_output(
        input_json=Path(args.input_json),
        output_json=Path(args.output_json),
    )
    summary = {
        "output_json": args.output_json,
        "frame_count": payload["frame_count"],
        "action_frame_count": payload["action_frame_count"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
