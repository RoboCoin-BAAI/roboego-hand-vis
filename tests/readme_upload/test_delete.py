import contextlib
import io
import unittest
from unittest.mock import Mock, patch
from roboego_hand_vis.upload_data.readme_upload import delete_from_hub as delete


class DeleteTests(unittest.TestCase):
    def test_preview_no_credentials_or_network(self):
        with patch.object(delete, 'make_delete_client') as client, patch.object(delete.getpass, 'getpass') as prompt:
            self.assertEqual(delete.main(['--hub','both','--repo-id','HumanCOIN/demo','--dry-run']), 0)
        client.assert_not_called()
        prompt.assert_not_called()

    def test_both_hubs_hidden_tokens_and_verified_deletion(self):
        clients = [Mock(), Mock()]
        for client in clients:
            client.repo_exists.return_value = False
        with patch.object(delete, 'get_token', return_value=None), patch.object(delete.getpass, 'getpass', side_effect=['hf-secret','ms-secret']), patch.object(delete, 'make_delete_client', side_effect=clients) as make:
            self.assertEqual(delete.main(['--hub','both','--repo-id','HumanCOIN/demo']), 0)
        self.assertEqual(make.call_args_list[0].args, ('huggingface', 'hf-secret'))
        self.assertEqual(make.call_args_list[1].args, ('modelscope', 'ms-secret'))
        for client in clients:
            client.delete_repo.assert_called_once_with(repo_id='HumanCOIN/demo', repo_type='dataset')

    def test_empty_second_token_prevents_all_deletions(self):
        client = Mock()
        with patch.object(delete, 'get_token', return_value=None), patch.object(delete.getpass, 'getpass', side_effect=['secret','']), patch.object(delete, 'make_delete_client', return_value=client):
            self.assertEqual(delete.main(['--hub','both','--repo-id','HumanCOIN/demo']), 1)
        client.delete_repo.assert_not_called()

    def test_failed_delete_not_retried_and_other_hub_continues(self):
        first, second = Mock(), Mock()
        first.delete_repo.side_effect = PermissionError('private-secret')
        second.repo_exists.return_value = False
        with patch.object(delete, 'get_token', return_value='private-secret'), patch.object(delete, 'make_delete_client', side_effect=[first,second]), contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(delete.main(['--hub','both','--repo-id','HumanCOIN/demo']), 1)
        self.assertNotIn('private-secret', output.getvalue())
        first.delete_repo.assert_called_once()
        second.delete_repo.assert_called_once()

    def test_existing_after_delete_is_not_success(self):
        client = Mock()
        client.repo_exists.return_value = True
        with patch.object(delete.time, 'sleep'), self.assertRaises(RuntimeError):
            delete.delete_and_verify(client, 'HumanCOIN/demo')
        client.delete_repo.assert_called_once()

    def test_reject_local_paths(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            delete.parse_args(['--hub','both','--repo-id','/home/user/dataset'])
