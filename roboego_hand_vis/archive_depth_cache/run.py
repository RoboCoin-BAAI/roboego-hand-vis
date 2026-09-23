#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from cli import config_from_yaml, main, parse_args, summarize_report_for_cli
from compute import DepthArchiveConfig, archive_depth_cache

__all__ = [
    "DepthArchiveConfig",
    "archive_depth_cache",
    "config_from_yaml",
    "main",
    "parse_args",
    "summarize_report_for_cli",
]


if __name__ == "__main__":
    main()
