#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = STAGE_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
for path in (STAGE_DIR, PACKAGE_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cli import (  # noqa: E402
    config_from_yaml,
    default_output_json_for_video,
    load_yaml,
    main,
    parse_args,
    run_video_simple_quality,
)

__all__ = [
    "config_from_yaml",
    "default_output_json_for_video",
    "load_yaml",
    "main",
    "parse_args",
    "run_video_simple_quality",
]


if __name__ == "__main__":
    main()
