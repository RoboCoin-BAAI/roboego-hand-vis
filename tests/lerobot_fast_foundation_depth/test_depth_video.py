from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from roboego_hand_vis.lerobot_fast_foundation_depth.L1.depth_video import HEVC12Writer, read_depth_video
from roboego_hand_vis.common.depth_io import depth_m_to_uint16_mm

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass


@unittest.skipUnless(FFMPEG, "FFmpeg with 12-bit libx265 required")
class DepthVideoTests(unittest.TestCase):
    def test_uses_current_ffmpeg_frame_sync_option(self):
        with tempfile.TemporaryDirectory() as directory:
            process = MagicMock()
            process.wait.return_value = 0
            with patch("roboego_hand_vis.lerobot_fast_foundation_depth.L1.depth_video.subprocess.Popen", return_value=process) as popen:
                writer = HEVC12Writer(Path(directory) / "depth.mkv", size=(64, 64), fps=30)
                writer.close()
            command = popen.call_args.args[0]
            self.assertNotIn("-vsync", command)
            self.assertEqual(command[command.index("-fps_mode") + 1], "cfr")

    def test_lossless_millimeter_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.mkv"
            first = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) / 1000.
            first[0, 0] = np.nan
            second = first[::-1].copy()
            second[0, :4] = [0., .001, 4., 4.095]
            encoder = HEVC12Writer(path, size=(64, 64), fps=30, ffmpeg=FFMPEG,
                                   max_depth_m=4.095, preset="ultrafast", threads=1)
            try:
                encoder.write(first)
                encoder.write(second)
            finally:
                encoder.close()
            decoded = list(read_depth_video(path, size=(64, 64), ffmpeg=FFMPEG))
            self.assertEqual(len(decoded), 2)
            for expected, actual in zip((first, second), decoded):
                np.testing.assert_array_equal(depth_m_to_uint16_mm(expected, max_depth_m=4.095),
                                              depth_m_to_uint16_mm(actual, max_depth_m=4.095))

    def test_range_guard(self):
        with self.assertRaises(ValueError):
            HEVC12Writer("unused.mkv", size=(64, 64), fps=30, max_depth_m=5.)


if __name__ == "__main__":
    unittest.main()
