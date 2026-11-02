import unittest
from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import crop_roi


class CropTests(unittest.TestCase):
    def test_crop_is_automatic_by_default(self):
        self.assertEqual(crop_roi((1920, 1200), (1920, 1088)), (0, 56, 1920, 1144))
        self.assertEqual(crop_roi((1920, 1088), (1920, 1088)), (0, 0, 1920, 1088))
        self.assertEqual(crop_roi((1921, 1201), (1920, 1088)), (0, 56, 1920, 1144))
        with self.assertRaises(ValueError):
            crop_roi((1920, 1200), (1920, 1088), False)
        self.assertEqual(crop_roi((1920, 1200), (1920, 1088), True), (0, 56, 1920, 1144))
        with self.assertRaises(ValueError):
            crop_roi((640, 480), (1920, 1088), True)
