#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hand.run_hamer_ego_filter import _write_refined_selection_preview  # noqa: E402
from roboego_hand_vis.egohand.candidate_refine import CandidateRefineConfig, refine_candidates_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact HaMeR selected-hand preview video.")
    parser.add_argument("--hamer_dir", required=True, help="Directory containing ego_hand_candidates_refined.json.")
    parser.add_argument("--manifest", required=True, help="Hand-window dataset_manifest.json.")
    parser.add_argument("--refined_json", default="", help="Optional explicit refined candidates JSON.")
    parser.add_argument("--preview_name", default="selection_preview.mp4")
    parser.add_argument("--contact_sheet_name", default="selection_contact_sheet.jpg")
    parser.add_argument("--semantic_json", default="", help="Frame-level Qwen checkpoint used when refined JSON must be built.")
    parser.add_argument("--stage1_time_offset_s", type=float, default=0.0)
    parser.add_argument("--refine_mode", default="strict_ego_audit")
    parser.add_argument("--preview_max_width", type=int, default=960)
    parser.add_argument("--write_frames", action="store_true", help="Also write per-frame refined jpgs for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hamer_dir = Path(args.hamer_dir).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    refined_json = Path(args.refined_json).expanduser().resolve() if args.refined_json else hamer_dir / "ego_hand_candidates_refined.json"
    if not refined_json.exists():
        source_json = hamer_dir / "ego_hand_candidates_overlay_filtered.json"
        if not source_json.exists():
            source_json = hamer_dir / "ego_hand_candidates.json"
        semantic_json = Path(args.semantic_json).expanduser().resolve() if args.semantic_json else None
        refine_candidates_file(
            candidates_json=source_json,
            stage1_actions_json=semantic_json,
            output_json=refined_json,
            config=CandidateRefineConfig(
                refine_mode=args.refine_mode,
                fps=30,
                stage1_time_offset_s=args.stage1_time_offset_s,
            ),
        )
    outputs = _write_refined_selection_preview(
        manifest=manifest,
        refined_json=refined_json,
        preview_path=hamer_dir / args.preview_name,
        contact_sheet_path=hamer_dir / args.contact_sheet_name,
        frames_dir=hamer_dir / "refined_annotated",
        write_frames=args.write_frames,
        preview_max_width=args.preview_max_width,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
