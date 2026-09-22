import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.common.depth_io import save_depth_uint16_png
from scripts.hand.export_final_rerun import iter_export_offsets, load_depth


def test_iter_export_offsets_uses_subset_when_requested():
    args = SimpleNamespace(start=10, num_frames=600, rrd_start=20, rrd_num_frames=30)

    offsets = list(iter_export_offsets(args))

    assert offsets[0] == (10, 20)
    assert offsets[-1] == (39, 49)
    assert len(offsets) == 30


def test_iter_export_offsets_defaults_to_all_processed_frames():
    args = SimpleNamespace(start=0, num_frames=3, rrd_start=None, rrd_num_frames=0)

    assert list(iter_export_offsets(args)) == [(0, 0), (1, 1), (2, 2)]


def test_load_depth_reads_uint16_png_archive(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    png_dir = depth_dir / "depth_uint16_png"
    expected = np.array([[0.25, 1.5], [np.nan, 2.0]], dtype=np.float32)
    assert save_depth_uint16_png(
        png_dir / "frame_000167_depth_mm.png",
        expected,
        max_depth_m=4.0,
    )

    depth = load_depth(depth_dir, 167)

    assert depth.dtype == np.float32
    assert np.allclose(depth[0], [0.25, 1.5])
    assert np.isnan(depth[1, 0])
    assert depth[1, 1] == np.float32(2.0)
