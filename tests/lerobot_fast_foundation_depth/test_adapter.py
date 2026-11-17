import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import sys

from roboego_hand_vis.lerobot_fast_foundation_depth.L1.compute import DEFAULT_CONFIG, episode_inputs, plan_session, run_session
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.run import main


class AdapterTests(unittest.TestCase):
    def test_default_quality_parameters(self):
        import yaml
        parameters = yaml.safe_load(DEFAULT_CONFIG.read_text())["parameters"]
        self.assertEqual(parameters["valid_iters"], 4)
        self.assertEqual(parameters["scale"], .5)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "session"
        self.ep = self.root / "data/chunk-000/episode_000000"
        self.ep.mkdir(parents=True)
        (self.root / "calibrations").mkdir()
        (self.root / "meta").mkdir()
        (self.root / "meta/info.json").write_text("{}")
        self.manifest = dict(reference_camera="left", videos={s: dict(path="/old/server/" + s) for s in ("left", "right")},
                             intrinsics={s: dict(fx=100., fy=100., cx=32., cy=24.) for s in ("left", "right")},
                             extrinsics=dict(baseline_m=.1))
        (self.root / "calibrations/dataset_manifest.json").write_text(json.dumps(self.manifest))
        for side in ("left", "right"):
            path = self.root / f"videos/chunk-000/observation.images.head_{side}_rectified/episode_000000.mp4"
            path.parent.mkdir(parents=True)
            path.touch()
        (self.ep / "observation.images.head_left_rectified.parquet").write_bytes(b"timestamp-bytes")

    def test_portable_paths_and_default_output(self):
        inputs = episode_inputs(self.root, self.ep)
        self.assertTrue(Path(inputs["manifest"]["videos"]["left"]["path"]).is_file())
        self.assertEqual(inputs["output"], self.root / "depth/chunk-000/episode_000000")

    def test_missing_rectified_video_fails(self):
        next((self.root / "videos").rglob("*.mp4")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing rectified video"):
            plan_session(self.root)

    def test_dry_run_does_not_write_depth(self):
        self.assertEqual(main(["--source", str(self.root), "--dry-run"]), 0)
        self.assertFalse((self.root / "depth").exists())

    @patch('roboego_hand_vis.common.stereo_quality.require_stereo_quality', side_effect=AssertionError('Runtime first-frame quality gate must not run'))
    def test_backend_adapter_copies_timestamps_and_overrides_range(self, quality_gate):
        config = self.root / "depth.yaml"
        config.write_text("paths: {}\nparameters:\n  num_frames: 600\n")
        model = self.root / "model"
        model.mkdir()
        checkpoint = model / "model.pth"
        checkpoint.touch()
        (model / "cfg.yaml").touch()
        captured = {}
        cli = ModuleType("roboego_hand_vis.fast_foundation_depth.cli")
        backend = ModuleType("roboego_hand_vis.fast_foundation_depth.compute")
        def config_from_yaml(settings, **kwargs):
            captured.update(settings)
            return SimpleNamespace(**kwargs)
        def infer(cfg):
            self.assertTrue(cfg.manifest.is_file())
            return dict(processed_frames=[0, 1], processed_frame_count=2)
        cli.config_from_yaml = config_from_yaml
        backend.run_fast_foundation_depth = infer
        with patch.dict(sys.modules, {cli.__name__: cli, backend.__name__: backend}):
            records = run_session(self.root, config=config, model_root=model, ckpt=checkpoint)
        self.assertEqual(captured["parameters"]["num_frames"], "auto")
        self.assertTrue(captured["parameters"]["overwrite"])
        output = Path(records[0]["output"])
        self.assertEqual((output / "source_timestamps.parquet").read_bytes(), b"timestamp-bytes")
        self.assertEqual(json.loads((output / "lerobot_depth.json").read_text())["processed_frames"], [0, 1])


if __name__ == "__main__":
    unittest.main()
