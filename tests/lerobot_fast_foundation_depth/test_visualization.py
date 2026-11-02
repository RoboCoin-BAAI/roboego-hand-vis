import unittest
import numpy as np
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.visualization import make_preview


class PreviewTests(unittest.TestCase):
    def test_two_meter_range_and_invalid_pixels(self):
        depth = np.array([[.01, .5, 1., 2., 3., 0., np.nan]], dtype=np.float32)
        before = depth.copy()
        left = np.full((1, 7, 3), 12, dtype=np.uint8)
        panel = make_preview(left, depth, near=0., far=2.)
        np.testing.assert_array_equal(panel[:, :7], left)
        np.testing.assert_array_equal(depth, before)
        colors = panel[:, 7:]
        np.testing.assert_array_equal(colors[0, 3], colors[0, 4])
        self.assertEqual(len({tuple(p) for p in colors[0, :4]}), 4)
        self.assertTrue(np.all(colors[0, 5:] == 0))

    def test_inverse_zero_range_rejected(self):
        with self.assertRaises(ValueError):
            make_preview(np.zeros((1, 1, 3), dtype=np.uint8), np.ones((1, 1)), near=0., mode="inverse")

    def test_default_range_stretches_near_depth_colors(self):
        left = np.zeros((1, 4, 3), dtype=np.uint8)
        depth = np.array([[.3, .4, .5, .6]], dtype=np.float32)
        colors = make_preview(left, depth)[:, 4:]
        wide = make_preview(left, depth, near=0., far=2.)[:, 4:]
        self.assertGreater(np.linalg.norm(colors[0, 0].astype(float) - colors[0, -1]),
                           np.linalg.norm(wide[0, 0].astype(float) - wide[0, -1]))
