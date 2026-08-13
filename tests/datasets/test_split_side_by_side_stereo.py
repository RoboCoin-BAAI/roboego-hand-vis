import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.split_side_by_side_stereo import split_frame_side_by_side


def test_split_frame_side_by_side_returns_left_and_right_halves():
    left = np.full((2, 3, 3), 10, dtype=np.uint8)
    right = np.full((2, 3, 3), 200, dtype=np.uint8)
    frame = np.concatenate([left, right], axis=1)

    got_left, got_right = split_frame_side_by_side(frame)

    assert np.array_equal(got_left, left)
    assert np.array_equal(got_right, right)


def test_split_frame_side_by_side_rejects_odd_width():
    frame = np.zeros((2, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="width must be even"):
        split_frame_side_by_side(frame)
