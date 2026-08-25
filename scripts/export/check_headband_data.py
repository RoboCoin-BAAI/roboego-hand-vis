#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.raw_headband_integrity import build_arg_parser, main


__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    main()
