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
    parser = argparse.ArgumentParser(description="Export a clean final hand result folder from temporal-gated hands.")
    parser.add_argument("--tracks_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temporal_annotated_dir", default="")
    parser.add_argument("--hamer_overlay_dir", default="")
    parser.add_argument("--source_frames_dir", default="")
    parser.add_argument("--copy_meshes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy_visuals", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_final_hand_result(
        tracks_json=Path(args.tracks_json).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        temporal_annotated_dir=Path(args.temporal_annotated_dir).expanduser() if args.temporal_annotated_dir else None,
        hamer_overlay_dir=Path(args.hamer_overlay_dir).expanduser() if args.hamer_overlay_dir else None,
        source_frames_dir=Path(args.source_frames_dir).expanduser() if args.source_frames_dir else None,
        copy_meshes=args.copy_meshes,
        copy_visuals=args.copy_visuals,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
