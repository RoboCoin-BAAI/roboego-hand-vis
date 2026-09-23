#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.qwen_annotation.frame_annotation import migrate_object_attribute_hand_side, validate_frame_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add object-level hand_side to existing Qwen frame annotation JSON files.")
    parser.add_argument("paths", nargs="+", help="JSON files or directories containing JSON files.")
    return parser.parse_args()


def iter_json_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
    return files


def migrate_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return False
    before = json.dumps(frames, ensure_ascii=False, sort_keys=True)
    migrate_object_attribute_hand_side(frames)
    validate_frame_annotations(frames)
    after = json.dumps(frames, ensure_ascii=False, sort_keys=True)
    if before == after:
        return False
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    changed = 0
    scanned = 0
    for path in iter_json_files(args.paths):
        scanned += 1
        if migrate_file(path):
            changed += 1
    print(json.dumps({"scanned": scanned, "changed": changed}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
