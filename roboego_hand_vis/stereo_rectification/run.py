#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from cli import main, parse_args, prepare_segment, prepare_stereo_unit
from compute import (
    _run_rectify,
    _selected_segment_dirs,
    _validate_segment,
    _write_manifest,
)

__all__ = [
    "_run_rectify",
    "_selected_segment_dirs",
    "_validate_segment",
    "_write_manifest",
    "main",
    "parse_args",
    "prepare_segment",
    "prepare_stereo_unit",
]


if __name__ == "__main__":
    main()

'''
cd /share/project/liuyou/humancoin/code/roboego-hand-vis

/share/project/liuyou/humancoin/envs/ego/bin/python \
  -m roboego_hand_vis.stereo_rectification.run \
  --session "/share/project/liuyou/humancoin/data/L1/batch_0000001/cp_0001_session_20261009_172052"
'''