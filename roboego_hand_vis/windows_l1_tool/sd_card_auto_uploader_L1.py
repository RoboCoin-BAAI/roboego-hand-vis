from __future__ import annotations

import sys

try:
    from .main import main
except ImportError:
    from main import main


if __name__ == "__main__":
    sys.exit(main())
