#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = STAGE_DIR.parent
for path in (STAGE_DIR, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cli import config_from_yaml, main, parse_args

__all__ = [
    "config_from_yaml",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
