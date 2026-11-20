import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from roboego_hand_vis.upload_data.readme_upload import upload2hub as upload


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'session'
        self.root.mkdir()
        for name in ['README.md', 'meta/info.json', 'data/a.parquet', 'data/a[0].parquet', 'backup/old', '.cache/blob', '.env']:
            p = self.root / name
            p.parent.mkdir(exist_ok=True, parents=True)
            p.write_text('{}')

    def test_ms_readme_strips_only_hf_gating_and_preserves_body(self):
        import yaml
        body = '# Example\n\nBody with gated: true stays here.\n'
        content = ('---\nlicense: cc-by-nc-4.0\nviewer: true\ntags: [HumanCOIN]\n'
                   'gated: true\nextra_gated_prompt: Apply\nextra_gated_fields:\n  Name: text\n'
                   'extra_gated_button_content: Agree\n---\n' + body)
        (self.root/'README.md').write_text(content)
        c = Mock()
        c.get_dataset.return_value = {'gated':True, 'gated_mode':'auto'}
        stages = []
        def inspect_upload(**kwargs):
            stage = Path(kwargs['folder_path'])
            stages.append(stage)
            prepared = (stage/'README.md').read_text()
            metadata = yaml.safe_load(prepared.split('---',2)[1])
            self.assertEqual(metadata, {'license':'cc-by-nc-4.0', 'viewer':True, 'tags':['HumanCOIN']})
            self.assertTrue(prepared.endswith(body))
            self.assertEqual((stage/'data/a.parquet').read_bytes(), (self.root/'data/a.parquet').read_bytes())
            self.assertEqual((self.root/'README.md').read_text(), content)
        c.upload_folder.side_effect = inspect_upload
        upload.upload_folder(c, hub='modelscope', folder=self.root, repo_id='HumanCOIN/session',
                             files=['README.md','data/a.parquet'])
        self.assertFalse(stages[0].exists())
        self.assertEqual((self.root/'README.md').read_text(), content)
        with upload.platform_folder(self.root, ['README.md'], 'huggingface') as folder:
            self.assertEqual(folder, self.root)
            self.assertEqual((folder/'README.md').read_text(), content)

    def test_ms_stage_cleanup_on_upload_failure(self):
        (self.root/'README.md').write_text('---\nlicense: cc-by-nc-4.0\ngated: true\n---\n# Body\n')
        stage = None
        with self.assertRaises(RuntimeError):
            with upload.platform_folder(self.root, ['README.md'], 'modelscope') as prepared:
                stage = prepared
                raise RuntimeError('failed')
        self.assertFalse(stage.exists())
        self.assertIn('gated: true', (self.root/'README.md').read_text())

    def test_invalid_card_fails_before_network_changes(self):
        (self.root/'README.md').write_text('---\ngated: [broken\n---\nBody')
        c = Mock()
        with self.assertRaises(Exception):
            upload.upload_folder(c, hub='modelscope', folder=self.root, repo_id='HumanCOIN/session',
                                 files=['README.md'], create_repo=True)
        self.assertEqual(c.mock_calls, [])

    def test_http_diagnostics_redact_credentials(self):
        exc = Exception('Do not show arbitrary exception contents')
        exc.response = Mock(status_code=403, headers={'x-request-id':'example-id'})
        exc.response.json.return_value = {'error':'Denied hf_exampletoken Bearer private-token custom-token'}
        result = upload.describe_error(exc, 'custom-token')
        self.assertIn('HTTP 403', result)
        self.assertIn('example-id', result)
        for secret in ('hf_exampletoken', 'private-token', 'custom-token'):
            self.assertNotIn(secret, result)

    def test_selection_and_readme(self):
        self.assertEqual(upload.plan_files(self.root), ['README.md', 'data/a.parquet', 'data/a[0].parquet', 'meta/info.json'])
        self.assertEqual(upload.plan_files(self.root, readme_only=True), ['README.md'])
        self.assertEqual(upload.plan_files(self.root, allow=['meta/*']), ['meta/info.json'])
        self.assertEqual(upload.plan_files(self.root, ignore=['data/*']), ['README.md', 'meta/info.json'])

    def test_dry_run_never_authenticates_or_uploads(self):
        with patch.object(upload, 'make_client') as client, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(upload.main(['--folders', str(self.root), '--dry-run']), 0)
            client.assert_not_called()
            self.assertIn('HumanCOIN/session', output.getvalue())
            self.assertNotIn('backup/old', output.getvalue())

    @unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "huggingface-hub is not installed")
    def test_hf_signature_and_existing_repo(self):
        from huggingface_hub import HfApi
        import inspect
        c = Mock()
        c.get_dataset.return_value = {'visibility':'gated', 'gated':True, 'gated_mode':'auto'}
        upload.upload_folder(c, hub='huggingface', folder=self.root,
                             repo_id='HumanCOIN/session', files=['README.md'])
        kwargs = c.upload_folder.call_args.kwargs
        inspect.signature(HfApi.upload_folder).bind(None, **kwargs)
        c.create_repo.assert_not_called()
        c.repo_exists.assert_not_called()
        self.assertNotIn('delete_patterns', kwargs)
        self.assertEqual(kwargs['allow_patterns'], ['README.md'])

    def test_ms_create_public(self):
        c = Mock()
        c.get_dataset.return_value = {'visibility':'gated', 'gated':True, 'gated_mode':'auto'}
        c.repo_exists.return_value = False
        upload.upload_folder(c, hub='modelscope', folder=self.root,
                             repo_id='HumanCOIN/session', files=['README.md'],
                             token='dummy', create_repo=True)
        c.create_repo.assert_called_once_with(repo_id='HumanCOIN/session', repo_type='dataset', visibility='public', license='cc-by-nc-4.0')
        self.assertEqual(c.upload_folder.call_args.kwargs['max_workers'], 8)
        c.update_repo_settings.assert_called_once_with(repo_id='HumanCOIN/session', repo_type='dataset', visibility='public', license='cc-by-nc-4.0')

    def test_ms_public_settings_applied_before_upload_and_failure_stops(self):
        c = Mock()
        c.get_dataset.return_value = {'visibility':'gated', 'gated':True, 'gated_mode':'auto'}
        upload.upload_folder(c, hub='modelscope', folder=self.root,
                             repo_id='HumanCOIN/session', files=['README.md'])
        names = [call[0] for call in c.mock_calls]
        self.assertLess(names.index('update_repo_settings'), names.index('upload_folder'))
        c.update_repo_settings.assert_called_once_with(repo_id='HumanCOIN/session', repo_type='dataset', visibility='public', license='cc-by-nc-4.0')
        self.assertNotIn('token', c.upload_folder.call_args.kwargs)
        c.reset_mock()
        c.update_repo_settings.side_effect = PermissionError()
        with self.assertRaises(PermissionError):
            upload.upload_folder(c, hub='modelscope', folder=self.root,
                                 repo_id='HumanCOIN/session', files=['README.md'])
        c.upload_folder.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("modelscope_hub"), "modelscope-hub is not installed")
    def test_actual_ms_sdk_public_settings_payload(self):
        from modelscope_hub import HubApi
        from modelscope_hub.constants import RepoType
        payload = HubApi._repo_settings_payload(RepoType.DATASET, {'visibility':'public'})
        self.assertEqual(payload, {'visibility':'public'})
        import inspect
        c = Mock()
        c.get_dataset.return_value = {'visibility':'gated', 'gated':True, 'gated_mode':'auto'}
        upload.upload_folder(c, hub='modelscope', folder=self.root,
                             repo_id='HumanCOIN/session', files=['README.md'], create_repo=True)
        inspect.signature(HubApi.repo_exists).bind(None, **c.repo_exists.call_args.kwargs)
        inspect.signature(HubApi.upload_folder).bind(None, **c.upload_folder.call_args.kwargs)

    def test_hf_create_public(self):
        c = Mock()
        c.get_dataset.return_value = {'visibility':'gated', 'gated':True, 'gated_mode':'auto'}
        c.repo_exists.return_value = False
        upload.upload_folder(c, hub='huggingface', folder=self.root,
                             repo_id='HumanCOIN/session', files=['README.md'], create_repo=True)
        c.create_repo.assert_called_once_with(repo_id='HumanCOIN/session', repo_type='dataset', token=None, private=False, exist_ok=True)

    def test_hf_gating_applied_before_file_upload(self):
        c = Mock()
        c.get_dataset.return_value = {'visibility':'gated', 'gated':True, 'gated_mode':'auto'}
        c.repo_exists.return_value = False
        upload.upload_folder(c, hub='huggingface', folder=self.root,
                             repo_id='HumanCOIN/session', files=['README.md'],
                             token='dummy', create_repo=True)
        c.update_repo_settings.assert_called_once_with(
            repo_id='HumanCOIN/session', repo_type='dataset', token='dummy', gated='auto')
        names = [call[0] for call in c.mock_calls]
        self.assertLess(names.index('create_repo'), names.index('update_repo_settings'))
        self.assertLess(names.index('update_repo_settings'), names.index('upload_folder'))

    def test_hf_gating_failure_prevents_upload(self):
        c = Mock()
        c.get_dataset.return_value = {'visibility':'gated', 'gated':True, 'gated_mode':'auto'}
        c.update_repo_settings.side_effect = PermissionError()
        with self.assertRaises(PermissionError):
            upload.upload_folder(c, hub='huggingface', folder=self.root,
                                 repo_id='HumanCOIN/session', files=['README.md'])
        c.upload_folder.assert_not_called()

    def test_permission_check_error_never_creates(self):
        c = Mock()
        c.get_dataset.return_value = {'visibility':'gated', 'gated':True, 'gated_mode':'auto'}
        c.repo_exists.side_effect = PermissionError()
        with self.assertRaises(PermissionError):
            upload.upload_folder(c, hub='modelscope', folder=self.root,
                                 repo_id='HumanCOIN/session', files=['README.md'], create_repo=True)
        c.create_repo.assert_not_called()
        c.upload_folder.assert_not_called()

    def test_retry_exhaustion_propagates(self):
        c = Mock()
        c.get_dataset.return_value = {'visibility':'gated', 'gated':True, 'gated_mode':'auto'}
        c.upload_folder.side_effect = ValueError('Expecting value: line 1 column 1 (char 0)')
        with patch.object(upload.time, 'sleep'), self.assertRaises(ValueError):
            upload.upload_folder(c, hub='modelscope', folder=self.root,
                                 repo_id='HumanCOIN/session', files=['README.md'], max_retries=2)
        self.assertEqual(c.upload_folder.call_count, 2)

    def test_symlinks_rejected(self):
        (self.root/'link').symlink_to(self.root/'README.md')
        with self.assertRaises(ValueError):
            upload.plan_files(self.root)

    def test_yaml_and_cli_override(self):
        cfg=Path(self.tmp.name)/'config.yaml'
        cfg.write_text(json.dumps({'hub':'ms','folders':[str(self.root)],'readme_only':True}))
        args=upload.parser_with_config(['--config',str(cfg),'--hub','hf'])
        self.assertEqual(args.hub,'huggingface')
        self.assertTrue(args.readme_only)
        cfg.write_text('create_repo: "false"')
        with self.assertRaises(ValueError):
            upload.parser_with_config(['--config',str(cfg)])

    def test_batch_failure_nonzero_and_continues(self):
        other=Path(self.tmp.name)/'other';other.mkdir();(other/'README.md').write_text('test')
        with patch.object(upload,'make_client'), patch.object(upload,'upload_folder',side_effect=[PermissionError('secret'),None]) as send, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            rc=upload.main(['--folders',str(self.root),str(other),'--skip-errors'])
        self.assertEqual(rc,1)
        self.assertEqual(send.call_count,2)
        self.assertNotIn('secret',errors.getvalue())


if __name__ == '__main__':
    unittest.main()
