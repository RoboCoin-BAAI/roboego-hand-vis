#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parents[2] / "roboego_hand_vis" / "vio"
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from prepare_entry import main


if __name__ == "__main__":
    main()
