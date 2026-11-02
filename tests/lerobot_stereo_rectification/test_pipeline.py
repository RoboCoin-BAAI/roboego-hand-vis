"""Small CPU integration test; no model or GPU required."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from roboego_hand_vis.common.stereo_manifest import load_foundation_inputs_from_manifest
from roboego_hand_vis.lerobot_headband_integrity.common import Context
from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import Encoder, rectify_episode, crop_output_maps, publish_files

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

@unittest.skipUnless(FFMPEG, "FFmpeg is needed for video integration testing")
class PipelineTests(unittest.TestCase):
    def test_encoded_pair_mapping_and_depth_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "session"
            episode = root / "data/chunk-000/episode_000000"
            episode.mkdir(parents=True)
            (root / "meta").mkdir()
            (root / "calibrations").mkdir()
            ffmpeg = FFMPEG
            features = {f"observation.images.head_{side}": dict(dtype="video", fps=30, shape=[48, 64, 3]) for side in ("left", "right")}
            (root / "meta/info.json").write_text(json.dumps(dict(fps=30, features=features)))
            calibration = {}
            for side, offset in (("left", 0), ("right", 1/30)):
                key = f"head_{side}"
                calibration[key] = dict(camera_model="pinhole", distortion_model="equidistant",
                    intrinsics=[32., 32., 32., 24.], distortion_coeffs=[0., 0., 0., 0.], resolution=[64, 48])
                pq.write_table(pa.table(dict(timestamp=[0., 0., .2] if side == "right" else [i / 30 for i in range(4)])), episode / f"observation.images.{key}.parquet")
                path = root / f"videos/chunk-000/observation.images.{key}/episode_000000.mp4"
                encoder = Encoder(path, (64, 48), 30, ffmpeg)
                try:
                    for i in range(3 if side == "right" else 4):
                        encoder.write(np.full((48, 64, 3), 30 + i * 40, dtype=np.uint8))
                finally:
                    encoder.close()
            transform = np.eye(4)
            transform[0, 3] = -.1
            calibration["head_right"].update(T_cn_cnm1=transform.tolist(), extrinsics_reference_camera="head_left")
            (root / "calibrations/camera-camchain.yaml").write_text(yaml.safe_dump(calibration))
            # Exercise raw Kalibr camera keys, as in the supplied dataset.
            raw = {"cam0": dict(calibration["head_left"], rostopic="/fisheye/left/image_raw/compressed"),
                   "cam1": dict(calibration["head_right"], rostopic="/fisheye/right/image_raw/compressed")}
            raw["cam1"].pop("extrinsics_reference_camera")
            (root / "calibrations/camera-camchain.yaml").write_text(yaml.safe_dump(raw))
            ctx = Context(root)
            originals = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*")
                         if p.is_file() and p != root / "meta/info.json"}
            plan = rectify_episode(ctx, episode, ffmpeg=ffmpeg, dry_run=True)
            self.assertEqual(plan["frame_policy"], "source_order")
            self.assertFalse((root / "stereo_rectification").exists())
            plan = rectify_episode(ctx, episode, ffmpeg=ffmpeg)
            out = Path(plan["calibration_output"])
            self.assertEqual(out, root / "calibrations")
            self.assertFalse((root / "cal").exists())
            mapping = pq.read_table(out / "frame_mapping.parquet").to_pydict()
            self.assertEqual(mapping["left_source_index"], [0, 1, 2, 3])
            self.assertEqual(mapping["right_source_index"], [0, 1, 2, None])
            self.assertEqual(plan["frame_counts"], {"left": 4, "right": 3})
            manifest = load_foundation_inputs_from_manifest(out / "dataset_manifest.json")
            self.assertAlmostEqual(manifest.baseline_m, .1)
            self.assertTrue(manifest.left_video.is_file())
            self.assertTrue(manifest.right_video.is_file())
            self.assertGreater(manifest.intrinsics["fx"], 0)
            video_mtime = manifest.left_video.stat().st_mtime_ns
            with patch.object(Encoder, "write", side_effect=AssertionError("must reuse videos")):
                reused = rectify_episode(ctx, episode, ffmpeg=ffmpeg)
            self.assertTrue(reused['skipped_existing'])
            self.assertEqual(manifest.left_video.stat().st_mtime_ns, video_mtime)
            # A moved dataset repairs the manifest without rewriting video bytes.
            manifest_path = out / 'dataset_manifest.json'
            moved = json.loads(manifest_path.read_text())
            moved['source_episode'] = '/old/location/episode'
            moved['frame_mapping'] = '/old/location/frame_mapping.parquet'
            moved['videos']['left']['path'] = '/old/location/left.mp4'
            manifest_path.write_text(json.dumps(moved))
            with patch.object(Encoder, "write", side_effect=AssertionError("must reuse videos")):
                self.assertTrue(rectify_episode(ctx, episode, ffmpeg=ffmpeg)['skipped_existing'])
            self.assertEqual(json.loads(manifest_path.read_text())['source_episode'], str(episode))
            self.assertEqual(manifest.left_video.stat().st_mtime_ns, video_mtime)
            with patch.object(Encoder, "write", side_effect=RuntimeError("force recompute")):
                with self.assertRaisesRegex(RuntimeError, "force recompute"):
                    rectify_episode(ctx, episode, ffmpeg=ffmpeg, skip_existing=False)
                with self.assertRaisesRegex(RuntimeError, "force recompute"):
                    rectify_episode(ctx, episode, ffmpeg=ffmpeg, balance=.5)
            params = yaml.safe_load((out / "rectify_params.yaml").read_text())
            self.assertEqual(params["resolution"], plan["output_resolution"])
            self.assertAlmostEqual(manifest.intrinsics["cx"], params["P1"][0][2])
            self.assertAlmostEqual(params["Q"][0][3], -params["P1"][0][2])
            cap = cv2.VideoCapture(str(manifest.left_video))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), params["resolution"][0])
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), params["resolution"][1])
            finally:
                cap.release()
            marker = out / "rectify_params.yaml"
            marker.write_text("old")
            with patch.object(Encoder, "write", side_effect=RuntimeError("encoding failure")):
                with self.assertRaisesRegex(RuntimeError, "encoding failure"):
                    rectify_episode(ctx, episode, ffmpeg=ffmpeg)
            self.assertEqual(marker.read_text(), "old")
            rectify_episode(ctx, episode, ffmpeg=ffmpeg)
            self.assertNotEqual(marker.read_text(), "old")
            info = json.loads((root / "meta/info.json").read_text())
            self.assertEqual(info["total_videos"], 4)
            for side in ("left", "right"):
                key = f"observation.images.head_{side}_rectified"
                self.assertEqual(info["features"][key]["shape"], [*reversed(params["resolution"]), 3])
                self.assertTrue((root / f"videos/chunk-000/{key}/episode_000000.mp4").is_file())
                self.assertEqual((episode / f"{key}.parquet").read_bytes(),
                                 (episode / f"observation.images.head_{side}.parquet").read_bytes())
            for relative, contents in originals.items():
                self.assertEqual((root / relative).read_bytes(), contents)
            self.assertFalse((root / "stereo_rectification").exists())
            self.assertTrue(manifest.left_video.is_file())

    def test_publication_failure_restores_features_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text("old-info")
            (root / "video.mp4").write_text("old-video")
            staging = root / ".staging"
            (staging / "meta").mkdir(parents=True)
            (staging / "meta/info.json").write_text("new-info")
            (staging / "video.mp4").write_text("new-video")
            rename = Path.rename
            def fail_metadata(source, target):
                if source == staging / "meta/info.json":
                    raise OSError("publish failure")
                return rename(source, target)
            with patch.object(Path, "rename", fail_metadata):
                with self.assertRaisesRegex(OSError, "publish failure"):
                    publish_files(staging, root)
            self.assertEqual((root / "meta/info.json").read_text(), "old-info")
            self.assertEqual((root / "video.mp4").read_text(), "old-video")

    def test_common_crop_updates_projection_and_q(self):
        maps = [(np.zeros((20, 30), dtype=np.float32), np.zeros((20, 30), dtype=np.float32)) for _ in range(2)]
        params = {"P1": np.array([[100., 0, 15, 0], [0, 100, 10, 0], [0, 0, 1, 0]]),
                  "P2": np.array([[100., 0, 15, -10], [0, 100, 10, 0], [0, 0, 1, 0]]),
                  "Q": np.array([[1., 0, 0, -15], [0, 1, 0, -10], [0, 0, 0, 100], [0, 0, 10, 0]])}
        with patch("roboego_hand_vis.stereo_rectification.rectify_video.compute_valid_crop_roi",
                   side_effect=[(2, 2, 29, 20), (4, 3, 28, 18)]):
            cropped, updated, size, roi = crop_output_maps(maps, params, (30, 20))
        self.assertEqual(roi, (4, 3, 28, 17))
        self.assertEqual(size, (24, 14))
        self.assertEqual(cropped[0][0].shape, (14, 24))
        self.assertEqual(updated["P1"][0, 2], 11)
        self.assertEqual(updated["P2"][1, 2], 7)
        self.assertEqual(updated["Q"][0, 3], -11)
        self.assertEqual(updated["Q"][1, 3], -7)
        self.assertEqual(params["P1"][0, 2], 15)


if __name__ == "__main__":
    unittest.main()
