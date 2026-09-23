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

from cli import config_from_yaml, load_yaml, main, parse_args  # noqa: E402

__all__ = ["config_from_yaml", "load_yaml", "main", "parse_args"]


if __name__ == "__main__":
    main()
