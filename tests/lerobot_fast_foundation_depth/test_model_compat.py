from types import SimpleNamespace
import unittest
from omegaconf import OmegaConf
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.model_compat import patch_model_config


class ModelCompatibilityTests(unittest.TestCase):
    def test_old_structured_checkpoint_uses_upstream_default(self):
        args = OmegaConf.create({"valid_iters": 8})
        OmegaConf.set_struct(args, True)
        cfg = OmegaConf.create({"valid_iters": 8})
        changes = patch_model_config(SimpleNamespace(args=args), cfg)
        self.assertEqual(changes, {"normalize": True})
        self.assertTrue(args.normalize)
        self.assertTrue(cfg.normalize)
        self.assertEqual(args.valid_iters, 8)

    def test_existing_false_is_preserved(self):
        args = OmegaConf.create({"normalize": False})
        self.assertEqual(patch_model_config(SimpleNamespace(args=args), {}), {})
        self.assertFalse(args.normalize)

    def test_adjacent_yaml_takes_precedence_over_default(self):
        args = SimpleNamespace()
        patch_model_config(SimpleNamespace(args=args), {"normalize": False})
        self.assertFalse(args.normalize)

    def test_invalid_config_is_not_silently_coerced(self):
        with self.assertRaises(ValueError):
            patch_model_config(SimpleNamespace(args={}), {"normalize": "false"})
