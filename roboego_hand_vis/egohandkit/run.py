#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from cli import config_from_yaml, main, parse_args
from compute import EgoHandKitConfig, build_egohandkit_command, run_egohandkit

__all__ = [
    "EgoHandKitConfig",
    "build_egohandkit_command",
    "config_from_yaml",
    "main",
    "parse_args",
    "run_egohandkit",
]


if __name__ == "__main__":
    main()

