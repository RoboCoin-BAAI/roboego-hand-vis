#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.final_hand_result import export_final_hand_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a compact final hand result folder.")
    parser.add_argument("--tracks_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--depth_anchored_json", default="")
    parser.add_argument("--depth_anchored_npz", default="")
    parser.add_argument("--depth_anchored_summary_json", default="")
    parser.add_argument("--copy_depth_anchored", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy_meshes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_final_hand_result(
        tracks_json=Path(args.tracks_json).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        depth_anchored_json=Path(args.depth_anchored_json).expanduser() if args.depth_anchored_json else None,
        depth_anchored_npz=Path(args.depth_anchored_npz).expanduser() if args.depth_anchored_npz else None,
        depth_anchored_summary_json=(
            Path(args.depth_anchored_summary_json).expanduser() if args.depth_anchored_summary_json else None
        ),
        copy_depth_anchored=args.copy_depth_anchored,
        copy_meshes=args.copy_meshes,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
