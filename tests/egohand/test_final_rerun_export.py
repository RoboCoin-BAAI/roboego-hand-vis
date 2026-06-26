import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hand.export_final_rerun import iter_export_offsets


def test_iter_export_offsets_uses_subset_when_requested():
    args = SimpleNamespace(start=10, num_frames=600, rrd_start=20, rrd_num_frames=30)

    offsets = list(iter_export_offsets(args))

    assert offsets[0] == (10, 20)
    assert offsets[-1] == (39, 49)
    assert len(offsets) == 30


def test_iter_export_offsets_defaults_to_all_processed_frames():
    args = SimpleNamespace(start=0, num_frames=3, rrd_start=None, rrd_num_frames=0)

    assert list(iter_export_offsets(args)) == [(0, 0), (1, 1), (2, 2)]
