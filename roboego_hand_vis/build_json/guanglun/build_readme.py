"""Compatibility entry point for the shared LeRobot readme builder."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from roboego_hand_vis.build_json.base.build_readme import (
    build, discover, infer_scene_type, lines, load, main,
)

if __name__ == '__main__':
    raise SystemExit(main())
