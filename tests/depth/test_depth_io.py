from pathlib import Path

import cv2
import numpy as np

from roboego_hand_vis.depth.io import depth_path_for_frame, load_depth_frame, save_depth_uint16_png


def test_load_depth_frame_prefers_npy_over_uint16_png(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    png_dir = depth_dir / "depth_uint16_png"
    png_dir.mkdir(parents=True)
    np.save(depth_dir / "frame_000003_foundation_depth_m.npy", np.array([[1.25]], dtype=np.float32))
    assert save_depth_uint16_png(png_dir / "frame_000003_depth_mm.png", np.array([[2.5]], dtype=np.float32), max_depth_m=4.0)

    depth, source = load_depth_frame(depth_dir, 3, return_path=True)

    assert source == depth_dir / "frame_000003_foundation_depth_m.npy"
    assert depth.dtype == np.float32
    assert depth.tolist() == [[1.25]]


def test_load_depth_frame_reads_uint16_png_as_meters(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    png_dir = depth_dir / "depth_uint16_png"
    png_dir.mkdir(parents=True)
    png = np.array([[0, 1250], [2500, 4000]], dtype=np.uint16)
    assert cv2.imwrite(str(png_dir / "frame_000004_depth_mm.png"), png)

    depth, source = load_depth_frame(depth_dir, 4, return_path=True)

    assert source == png_dir / "frame_000004_depth_mm.png"
    assert depth.dtype == np.float32
    assert np.isnan(depth[0, 0])
    assert depth[0, 1] == 1.25
    assert depth[1, 0] == 2.5
    assert depth[1, 1] == 4.0


def test_depth_path_for_frame_returns_legacy_npy_name():
    path = depth_path_for_frame(Path("depth"), 7)

    assert path == Path("depth/frame_000007_foundation_depth_m.npy")
