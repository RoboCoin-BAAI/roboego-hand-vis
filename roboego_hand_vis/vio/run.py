from __future__ import annotations

from pathlib import Path
import sys

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from cli import config_from_yaml_and_args, load_yaml, main, parse_args
from runner import OpenVinsRunConfig, build_openvins_command, run_openvins

__all__ = [
    "OpenVinsRunConfig",
    "build_openvins_command",
    "config_from_yaml_and_args",
    "load_yaml",
    "main",
    "parse_args",
    "run_openvins",
]


if __name__ == "__main__":
    main()
