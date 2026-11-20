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
spec = importlib.util.spec_from_file_location('object_cleaner_test_module', DIRECTORY / 'clean_object_names_qwen.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VOCAB = {row['name'] for row in json.loads((DIRECTORY / 'object_vocabulary.json').read_text(encoding='utf-8'))['entries']}


class ObjectNameCleaningTests(unittest.TestCase):
    def test_only_object_names_change_including_repeated_names_and_unresolved(self):
        raw = (b'\xef\xbb\xbf{\r\n"objects": [ '
               b'{"name":"two red cups","attributes":{"color":"red","name":"nested"},"location":"left"}, '
               b'{"name":"blue cup","attributes":{}}, '
               b'{"name":"unidentified part","attributes":{}} ], '
               b'"subtask":[{"object":"two red cups","destination":"blue cup"}], '
               b'"task_description":"two red cups", "unchanged":1.00\r\n}\r\n')
        items = [dict(index=0, name='cup', reason='Remove count and color'),
                 dict(index=1, name='cup', reason='Remove color'),
                 dict(index=2, name=None, reason='Unclear identity')]
        revised = module.patch_names(raw, items)
        self.assertEqual(revised, raw.replace(b'"name":"two red cups"', b'"name":"cup"').replace(
            b'"name":"blue cup"', b'"name":"cup"'))
        data = json.loads(revised.decode('utf-8-sig'))
        self.assertEqual(len(data['objects']), 3)
        self.assertEqual(data['subtask'][0]['object'], 'two red cups')
        self.assertEqual(data['objects'][0]['attributes']['name'], 'nested')

    def test_name_policy_rejects_modifiers_count_reordering_and_retains_compound_nouns(self):
        for name in ['cup', 'washing machine', 'remote control', 'first aid kit', 'cotton swab', 'scissors', 'glass', 'orange']:
            module.validate_items(dict(items=[dict(index=0, name=name, reason='Object category')]), ['source'], VOCAB)
        for name in ['two cups', '2 cups', 'a pair of scissors', 'red cup', 'large box', 'small', 'cups', 'plastic cup']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.validate_items(dict(items=[dict(index=0, name=name, reason='Invalid')]), ['source'], VOCAB)
        with self.assertRaises(ValueError):
            module.validate_items(dict(items=[]), ['source'], VOCAB)
        with self.assertRaises(ValueError):
            module.validate_items(dict(items=[dict(index=1, name='cup', reason='Wrong index')]), ['source'], VOCAB)

    def fixture(self, temporary):
        root = Path(temporary) / 'data'
        path = root / 'batch_0000001/sample/annotation/episode_000000_structured.json'
        path.parent.mkdir(parents=True)
        raw = b'{"objects":[{"name":"two red cups","attributes":{"color":"red"}}],"task_description":"Move cups","subtask":[{"object":"two red cups"}]}'
        path.write_bytes(raw)
        args = SimpleNamespace(root=root, report_dir=Path(temporary) / 'reports', dry_run=False,
            apply=False, retry_failed=False, model='qwen3.7-plus', max_tokens=4096, timeout=30)
        endpoint = SimpleNamespace(settings=lambda model: ('secret', 'https://example.invalid', 'mock'))
        payload = dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(
            dict(items=[dict(index=0, name='cup', reason='Remove quantity and color')]))))])
        return path, raw, args, endpoint, payload

    def test_cached_result_is_not_rebilled_before_or_after_apply(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, raw, args, endpoint, payload = self.fixture(temporary)
            with patch.object(module, 'call_model', return_value=(payload, 'mock')) as request:
                proposed = module.process(path, args, endpoint, VOCAB)
                self.assertEqual(proposed['changed_names'], 1)
                self.assertEqual(path.read_bytes(), raw)
                args.apply = True
                applied = module.process(path, args, endpoint, VOCAB)
                self.assertTrue(applied['applied'])
                again = module.process(path, args, endpoint, VOCAB)
                self.assertFalse(again['applied'])
                self.assertEqual(request.call_count, 1)
            self.assertEqual(path.read_bytes(), raw.replace(b'"name":"two red cups"', b'"name":"cup"'))
            self.assertEqual((Path(applied['state']) / 'annotation_before.json').read_bytes(), raw)

    def test_unknown_api_outcome_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, raw, args, endpoint, _ = self.fixture(temporary)
            args.apply = True
            with patch.object(module, 'call_model', side_effect=TimeoutError('unknown')) as request:
                with self.assertRaises(TimeoutError):
                    module.process(path, args, endpoint, VOCAB)
                with self.assertRaisesRegex(RuntimeError, 'may have been billed'):
                    module.process(path, args, endpoint, VOCAB)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(path.read_bytes(), raw)

    def test_dry_run_never_calls_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, raw, args, endpoint, _ = self.fixture(temporary)
            args.dry_run = True
            with patch.object(module, 'call_model') as request:
                result = module.process(path, args, endpoint, VOCAB)
            self.assertEqual(result['status'], 'prepared')
            request.assert_not_called()
            self.assertEqual(path.read_bytes(), raw)

    def test_invalid_name_is_retained_for_review_without_losing_other_mappings(self):
        items = [dict(index=0, name='rubber band', reason='Material modifier'),
                 dict(index=1, name='cup', reason='Canonical'), dict(index=2, name='box')]
        payload = dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(items)))])
        result = module.parse_response(payload, ['rubber band', 'red cup', 'box'], VOCAB)
        self.assertIsNone(result[0]['name'])
        self.assertEqual(result[1]['name'], 'cup')
        self.assertIsNone(result[2]['name'])
        items[1]['index'] = 0
        payload['choices'][0]['message']['content'] = json.dumps(items)
        with self.assertRaises(ValueError):
            module.parse_response(payload, ['rubber band', 'red cup', 'box'], VOCAB)

    def test_legacy_cached_object_response_is_reused_after_parser_fix(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, args, endpoint, payload = self.fixture(temporary)
            names = ['two red cups']
            old_policy = module.digest(dict(model=args.model, max_tokens=args.max_tokens, context='Move cups',
                prompt=module.PROMPT, vocabulary=sorted(VOCAB), code=module.LEGACY_PROCESSORS[0]))
            old_folder = args.report_dir / 'jobs' / module.digest(str(path))[:24] / old_policy[:24] / module.digest(names)[:24]
            attempt = old_folder / 'attempt-0001'
            attempt.mkdir(parents=True)
            (attempt / 'response.json').write_text(json.dumps(payload))
            with patch.object(module, 'call_model') as request:
                result = module.process(path, args, endpoint, VOCAB)
            request.assert_not_called()
            self.assertEqual(result['changed_names'], 1)
            self.assertEqual(Path(result['state']), old_folder)


if __name__ == '__main__':
    unittest.main()
