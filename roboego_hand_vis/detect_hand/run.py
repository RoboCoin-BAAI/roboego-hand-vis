#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = STAGE_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT, STAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from .cli import (  # noqa: E402
        DetectHandConfig,
        config_from_yaml,
        load_yaml,
        main,
        parse_args,
        process_video,
        run_detect_hand_stage,
    )
except ImportError:
    from cli import (  # noqa: E402
        DetectHandConfig,
        config_from_yaml,
        load_yaml,
        main,
        parse_args,
        process_video,
        run_detect_hand_stage,
    )


__all__ = [
    "DetectHandConfig",
    "config_from_yaml",
    "load_yaml",
    "main",
    "parse_args",
    "process_video",
    "run_detect_hand_stage",
]


if __name__ == "__main__":
    main()
