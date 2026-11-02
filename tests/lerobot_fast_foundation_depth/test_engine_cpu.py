from unittest.mock import patch
"""Real CPU decode/save pipeline, with model disparity injected instead of CUDA."""
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from roboego_hand_vis.lerobot_fast_foundation_depth.L1.engine import DepthEngine, _is_unsupported_grid_sample_batch


@dataclass
class Config:
    start: int = 0
    num_frames: int | None = None
    scale: float = .5
    valid_iters: int = 8
    min_disp: float = 1e-3
    max_depth_m: float = 4.
    save_npy: bool = False
    save_depth_png: bool = True
    save_vis: bool = False
    preview_every_n_frames: int = 5
    manifest: Path = Path()
    output_dir: Path = Path()


class EngineCPUTests(unittest.TestCase):
    def test_only_recognizes_unsupported_grid_sample_batch_errors(self):
        self.assertTrue(_is_unsupported_grid_sample_batch(RuntimeError(
            "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED for non-contiguous input")))
        self.assertFalse(_is_unsupported_grid_sample_batch(RuntimeError("CUDA out of memory")))

    @patch('roboego_hand_vis.common.stereo_quality.require_stereo_quality')
    def test_decode_preprocess_save_and_depth_math(self, quality_gate):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "data/chunk-000/episode_000000"
            episode.mkdir(parents=True)
            (root / "calibrations").mkdir()
            (episode / "observation.images.head_left_rectified.parquet").write_bytes(b"timestamps")
            manifest = dict(reference_camera="left", videos={side: {} for side in ("left", "right")},
                intrinsics={side: dict(fx=32., fy=32., cx=32., cy=24.) for side in ("left", "right")},
                extrinsics=dict(baseline_m=.1))
            (root / "calibrations/dataset_manifest.json").write_text(json.dumps(manifest))
            for side in ("left", "right"):
                video = root / f"videos/chunk-000/observation.images.head_{side}_rectified/episode_000000.mp4"
                video.parent.mkdir(parents=True)
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30., (64, 48))
                self.assertTrue(writer.isOpened())
                for i in range(4):
                    writer.write(np.full((48, 64, 3), 50 + i, dtype=np.uint8))
                writer.release()
            # Execute the unchanged original depth conversion function directly,
            # without importing its Torch/model dependencies on this CPU host.
            original = Path(__file__).resolve().parents[2] / "roboego_hand_vis/fast_foundation_depth/compute.py"
            tree = ast.parse(original.read_text(encoding="utf-8"))
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "disparity_to_depth_original_size")
            namespace = dict(np=np, cv2=cv2)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(original), "exec"), namespace)
            engine = DepthEngine.__new__(DepthEngine)
            engine.cv2, engine.torch = cv2, None
            engine.config, engine.model_args = Config(), {}
            engine.options = dict(prefetch=2, write_queue=2, cpu_threads=1, batch_size=3)
            engine.backend = SimpleNamespace(disparity_to_depth_original_size=namespace["disparity_to_depth_original_size"], to_jsonable=lambda x: x)
            batches = []
            def infer_batch(pairs):
                batches.append(len(pairs))
                for left, right in pairs:
                    self.assertEqual(left.shape, right.shape)
                return [np.full(left.shape[:2], 2., dtype=np.float32) for left, _ in pairs], .001
            engine.infer_batch = infer_batch
            with patch('roboego_hand_vis.common.stereo_quality.require_stereo_quality'):
                report = engine.process(dict(session=str(root), episode=str(episode)))
            self.assertEqual(report["frames"], 4)
            self.assertEqual(batches, [3, 1])
            output = Path(report["output"])
            files = list((output / "depth_uint16_png").glob("*.png"))
            self.assertEqual(len(files), 4)
            depth = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
            self.assertEqual(depth.dtype, np.uint16)
            self.assertTrue(np.all(depth == 800))
            metadata = json.loads((output / "lerobot_depth.json").read_text())
            self.assertEqual(metadata["scale"], .5)
            self.assertEqual(metadata["valid_iters"], 8)
            self.assertEqual(metadata["batch_size"], 3)
            self.assertEqual(metadata["processed_frames"], [0, 1, 2, 3])
            self.assertGreater(metadata["fps"], 0)
            engine.options["right_frame_offset"] = -1
            batches.clear()
            shifted = engine.process(dict(session=str(root), episode=str(episode)))
            self.assertEqual(shifted["frames"], 3)
            self.assertEqual(batches, [3])
            self.assertEqual(Path(shifted["output"]).name, "episode_000000_right_offset_m1")
            shifted_meta = json.loads((Path(shifted["output"]) / "lerobot_depth.json").read_text())
            self.assertEqual(shifted_meta["processed_frames"], [1, 2, 3])
            self.assertEqual(shifted_meta["right_frame_offset"], -1)
            self.assertEqual(len(list((output / "depth_uint16_png").glob("*.png"))), 4)
            engine.options["right_frame_offset"] = 0
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                try:
                    import imageio_ffmpeg
                    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                except ImportError:
                    return
            from roboego_hand_vis.lerobot_fast_foundation_depth.L1.depth_video import read_depth_video
            engine.options.update(depth_format="hevc12", ffmpeg=ffmpeg, hevc12_preset="ultrafast", encode_threads=1)
            engine.process(dict(session=str(root), episode=str(episode)))
            self.assertFalse((output / "depth_uint16_png").exists())
            decoded = list(read_depth_video(output / "depth_hevc12.mkv", size=(64, 48), ffmpeg=ffmpeg))
            self.assertEqual(len(decoded), 4)
            np.testing.assert_allclose(decoded[0], .8)
            video_metadata = json.loads((output / "lerobot_depth.json").read_text())["depth_storage"]
            self.assertTrue(video_metadata["lossless"])
            self.assertEqual(video_metadata["frames"], 4)


if __name__ == "__main__":
    unittest.main()
