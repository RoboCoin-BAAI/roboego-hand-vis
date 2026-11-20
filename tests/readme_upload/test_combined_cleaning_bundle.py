import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[2] / 'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('combined_cleaning_test_module', DIRECTORY / 'clean_names_and_actions_qwen.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CombinedBundleTests(unittest.TestCase):
    def fixture(self, temporary):
        root = Path(temporary) / 'data'
        path = root / 'batch_0000001/sample/annotation/episode_000000_structured.json'
        path.parent.mkdir(parents=True)
        raw = (b'{\r\n"objects":[{"name":"red cup","attributes":{"color":"red"}}],'
               b'"actions":{"vocabulary":["trim","retrim"]},"subtask":[{"action":["retrim"],"object":"red cup"}],'
               b'"task_description":"task","scene_classification":{"category":"other","area":"other"},"keep":1.00\r\n}')
        path.write_bytes(raw)
        before = module.source_fields(json.loads(raw))
        record = dict(path=path.relative_to(root).as_posix(), status='done', before=before,
                      after=dict(object_names=['cup'], action_vocabulary=['trim']))
        return root, path, raw, record

    def test_apply_preserves_the_latest_scene_and_other_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, path, raw, record = self.fixture(temporary)
            scene_updated = raw.replace(b'"category":"other","area":"other"', b'"category":"household","area":"kitchen"')
            path.write_bytes(scene_updated)
            result = module.apply_record(root, record, Path(temporary) / 'reports')
            self.assertTrue(result['changed'])
            expected = scene_updated.replace(b'"name":"red cup"', b'"name":"cup"').replace(b'["trim","retrim"]', b'["trim"]')
            self.assertEqual(path.read_bytes(), expected)
            self.assertFalse(module.apply_record(root, record, Path(temporary) / 'reports')['changed'])

    def test_target_field_conflict_blocks_write_instead_of_guessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, path, raw, record = self.fixture(temporary)
            changed = raw.replace(b'"name":"red cup"', b'"name":"bottle"')
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, 'Source labels changed'):
                module.apply_record(root, record, Path(temporary) / 'reports')
            self.assertEqual(path.read_bytes(), changed)

    def test_prepare_uses_both_cleaners_without_applying(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, path, raw, _ = self.fixture(temporary)
            args = SimpleNamespace(root=root, state_dir=Path(temporary) / 'state', apply=False, dry_run=False)
            def object_process(path, args, endpoint, vocabulary):
                self.assertFalse(args.apply)
                return dict(changes=[dict(index=0, before='red cup', after='cup', reason='Remove color')])
            def action_process(path, args, endpoint, vocabulary):
                self.assertFalse(args.apply)
                # Simulate scene completion while models are being processed.
                path.write_bytes(raw.replace(b'"area":"other"', b'"area":"kitchen"'))
                return dict(after=['trim'])
            with patch.object(module.objects, 'process', side_effect=object_process), patch.object(module.actions, 'process', side_effect=action_process):
                result = module.prepare_one(path, args, None, set(), {})
            self.assertEqual(result['after'], dict(object_names=['cup'], action_vocabulary=['trim']))
            self.assertIn(b'"name":"red cup"', path.read_bytes())
            self.assertIn(b'["trim","retrim"]', path.read_bytes())

    def test_preview_bundle_cannot_be_applied(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / 'bundle.json'
            bundle.write_text(json.dumps(dict(version=1, status='preview', scope=['objects[].name', 'actions.vocabulary'])))
            with self.assertRaisesRegex(ValueError, 'completed replacement bundle'):
                module.apply_bundle(SimpleNamespace(bundle=bundle))


if __name__ == '__main__':
    unittest.main()
