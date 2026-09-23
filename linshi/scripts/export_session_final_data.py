#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from roboego_hand_vis.export.session_final_data import export_session_final_data, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a compact final session dataset.")
    parser.add_argument("--session_dir", required=True)
    parser.add_argument("--output_root", default="final_data")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = export_session_final_data(
        session_dir=Path(args.session_dir),
        output_root=Path(args.output_root),
        overwrite=args.overwrite,
        fps=args.fps,
    )
    print(f"Output: {summary['output_session']}")
    print(f"Segments: {len(summary['segments'])}")


if __name__ == "__main__":
    main()
