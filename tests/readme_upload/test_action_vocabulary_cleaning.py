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
spec = importlib.util.spec_from_file_location('action_cleaner_test_module', DIRECTORY / 'clean_action_vocabulary_qwen.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TABLE = json.loads((DIRECTORY / 'action_vocabulary.json').read_text(encoding='utf-8'))
REFERENCE = {key: [entry['verb'] for entry in entries] for key, entries in TABLE['groups'].items()}
VOCAB = {verb for verbs in REFERENCE.values() for verb in verbs}


class ActionVocabularyTests(unittest.TestCase):
    def test_lexical_re_verbs_are_not_treated_as_repeated_execution(self):
        names = ['return', 'replace', 'recover', 'repair']
        items = [dict(index=i, name=name, reason='Distinct lexical verb') for i, name in enumerate(names)]
        self.assertEqual(module.validate_items(dict(items=items), names, {'turn', 'place', 'cover', 'pair'}), items)

    def test_table_excludes_numeric_footer_and_preserves_valid_reference_verbs(self):
        self.assertEqual(TABLE['counts'], {'Subtask_100': 99, 'Atomic_50': 46})
        self.assertNotIn('110', VOCAB)
        self.assertTrue({'trim', 'check', 'adjust', 'remove', 'retract', 'release'} <= VOCAB)

    def test_only_vocabulary_changes_subtasks_text_and_all_other_bytes_are_retained(self):
        raw = (b'\xef\xbb\xbf{\r\n"actions":{"vocabulary":["trim", "check", "adjust", "retrim"],"other":1.00},'
               b'"subtask":[{"action":["retrim"],"text":"Retrim hair","start_s":1.00}],'
               b'"objects":[{"name":"red cup"}],"task_description":"Retrim hair"\r\n}\r\n')
        names = ['trim', 'check', 'adjust', 'retrim']
        items = [dict(index=i, name='trim' if name == 'retrim' else name, reason='Normalize') for i, name in enumerate(names)]
        normalized = module.normalized_names(names, module.validate_items(dict(items=items), names, VOCAB))
        self.assertEqual(normalized, ['trim', 'check', 'adjust'])
        result = module.patch_vocabulary(raw, normalized)
        self.assertEqual(result, raw.replace(b'["trim", "check", "adjust", "retrim"]', b'["trim", "check", "adjust"]'))
        self.assertEqual(json.loads(result.decode('utf-8-sig'))['subtask'][0]['action'], ['retrim'])

    def test_modifiers_phrases_and_missing_indices_rejected_but_opposites_distinct(self):
        for verb in ['carefully wipe', 'wipe_table', 'quickly', 'red', 'retrim']:
            with self.subTest(verb=verb), self.assertRaises(ValueError):
                module.validate_items(dict(items=[dict(index=0, name=verb, reason='x')]), ['source'], VOCAB)
        names = ['open', 'close', 'fold', 'unfold', 'screw', 'unscrew', 'remove', 'move']
        items = [dict(index=i, name=verb, reason='Distinct action') for i, verb in enumerate(names)]
        module.validate_items(dict(items=items), names, VOCAB)
        self.assertEqual(module.normalized_names(names, items), names)
        with self.assertRaises(ValueError):
            module.validate_items(dict(items=[]), ['source'], VOCAB)

    def test_apply_deduplication_is_cached_without_rebilling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'data'
            path = root / 'batch_0000001/sample/annotation/episode_000000_structured.json'
            path.parent.mkdir(parents=True)
            raw = b'{"actions":{"vocabulary":["trim","retrim"]},"subtask":[{"action":["retrim"]}],"task_description":"Cut hair"}'
            path.write_bytes(raw)
            args = SimpleNamespace(root=root, report_dir=Path(temporary) / 'reports', dry_run=False,
                apply=False, model='qwen3.7-plus', max_tokens=4096, timeout=30, retry_failed=False)
            endpoint = SimpleNamespace(settings=lambda model: ('secret', 'https://example.invalid', 'mock'))
            payload = dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(dict(
                items=[dict(index=i, name='trim', reason='Equivalent trimming') for i in range(2)]))))])
            with patch.object(module, 'request_model', return_value=payload) as request:
                proposed = module.process(path, args, endpoint, REFERENCE)
                self.assertEqual(proposed['after'], ['trim'])
                self.assertEqual(path.read_bytes(), raw)
                args.apply = True
                self.assertTrue(module.process(path, args, endpoint, REFERENCE)['applied'])
                self.assertFalse(module.process(path, args, endpoint, REFERENCE)['applied'])
                self.assertEqual(request.call_count, 1)
            self.assertEqual(path.read_bytes(), raw.replace(b'["trim","retrim"]', b'["trim"]'))

    def test_unresolved_labels_are_retained_and_deduplication_is_stable(self):
        names = ['pick up', 'mystery phrase', 'pick']
        items = [dict(index=0, name='pick', reason='Canonical'),
                 dict(index=1, name=None, reason='Unclear'), dict(index=2, name='pick', reason='Already canonical')]
        module.validate_items(dict(items=items), names, VOCAB)
        self.assertEqual(module.normalized_names(names, items), ['pick', 'mystery phrase'])

    def test_top_level_array_envelope_is_accepted_but_missing_entries_are_not(self):
        names = ['trim', 'retrim']
        items = [dict(index=i, name='trim', reason='Same action') for i in range(2)]
        payload = dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(items)))])
        self.assertEqual(module.parse_response(payload, names, VOCAB), items)
        payload['choices'][0]['message']['content'] = json.dumps(items[:1])
        with self.assertRaises(ValueError):
            module.parse_response(payload, names, VOCAB)

    def test_old_code_hash_cached_array_response_is_recovered_without_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'data'
            path = root / 'batch_0000001/sample/annotation/episode_000000_structured.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(actions=dict(vocabulary=['trim', 'retrim']), task_description='Cut hair')))
            args = SimpleNamespace(root=root, report_dir=Path(temporary) / 'reports', dry_run=False,
                apply=False, model='qwen3.7-plus', max_tokens=4096, timeout=30, retry_failed=False)
            old_policy = module.digest(dict(model=args.model, max_tokens=args.max_tokens, context='Cut hair',
                reference=REFERENCE, prompt=module.PROMPT, code=module.LEGACY_PROCESSORS[0]))
            old_folder = args.report_dir / 'jobs' / module.digest(str(path))[:24] / old_policy[:24] / module.digest(['trim', 'retrim'])[:24]
            attempt = old_folder / 'attempt-0001'
            attempt.mkdir(parents=True)
            payload = dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(
                [dict(index=i, name='trim', reason='Equivalent') for i in range(2)])))])
            (attempt / 'response.json').write_text(json.dumps(payload))
            with patch.object(module, 'request_model') as request:
                result = module.process(path, args, None, REFERENCE)
            request.assert_not_called()
            self.assertEqual(result['after'], ['trim'])
            self.assertEqual(Path(result['state']), old_folder)


if __name__ == '__main__':
    unittest.main()
