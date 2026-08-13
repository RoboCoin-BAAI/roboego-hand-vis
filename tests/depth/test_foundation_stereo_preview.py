import numpy as np

from scripts.run_foundation_stereo_video import make_depth_preview_panel_rgb


def test_make_depth_preview_panel_rgb_preserves_left_rgb_channels():
    left_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    left_rgb[:, :] = [255, 0, 0]
    depth_m = np.full((2, 3), 1.0, dtype=np.float32)

    panel = make_depth_preview_panel_rgb(left_rgb, depth_m, max_depth_m=4.0)

    assert panel.shape == (2, 6, 3)
    assert np.array_equal(panel[:, :3], left_rgb)
