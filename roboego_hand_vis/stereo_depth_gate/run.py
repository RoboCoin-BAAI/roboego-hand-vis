#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from cli import (  # noqa: E402
    StereoDepthGateRunConfig,
    config_from_yaml_and_args,
    load_yaml,
    main,
    parse_args,
    run_stereo_depth_gate,
)

__all__ = [
    "StereoDepthGateRunConfig",
    "config_from_yaml_and_args",
    "load_yaml",
    "main",
    "parse_args",
    "run_stereo_depth_gate",
]


if __name__ == "__main__":
    main()
