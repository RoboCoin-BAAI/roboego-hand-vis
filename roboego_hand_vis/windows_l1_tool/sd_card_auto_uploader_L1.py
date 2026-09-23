from __future__ import annotations

import sys
from pathlib import Path

try:
    from roboego_hand_vis.windows_l1_tool.main import main
except ModuleNotFoundError:
    TOOL_DIR = Path(__file__).resolve().parent
    if str(TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(TOOL_DIR))
    from main import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
