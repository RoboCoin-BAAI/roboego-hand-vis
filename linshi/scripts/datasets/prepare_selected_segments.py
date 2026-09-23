#!/usr/bin/env python3
from __future__ import annotations

import sys

from roboego_hand_vis.stereo_rectification import compute


if __name__ == "__main__":
    compute.main()
else:
    sys.modules[__name__] = compute
