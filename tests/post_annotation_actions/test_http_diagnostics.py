import io
import json
import urllib.error
import urllib.request

import pytest

from roboego_hand_vis.post_annotation_actions.text_resolver import call_qwen_text_resolver
from roboego_hand_vis.post_annotation_actions.model_subtasks import call_qwen_subtask_resolver


@pytest.mark.parametrize('body', [
    json.dumps({'error': {'code': 'ModelNotFound', 'message': 'The model qwen-plus does not exist.'}}),
    '<html>proxy route not found</html>',
])
def test_text_http_error_keeps_server_body_and_context(monkeypatch, body):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'private-test-key')
    def fail(req, **kwargs):
        raise urllib.error.HTTPError(req.full_url, 404, 'Not Found', {}, io.BytesIO(body.encode()))
    monkeypatch.setattr(urllib.request, 'urlopen', fail)
    with pytest.raises(urllib.error.HTTPError) as caught:
        call_qwen_text_resolver(review_payload={'frames': []}, model='qwen-plus', base_url='https://example.test/v1/chat/completions')
    message = str(caught.value)
    assert '404' in message and 'text_resolver' in message and 'qwen-plus' in message
    assert 'https://example.test/v1/chat/completions' in message
    assert body in message
    assert 'private-test-key' not in message


def test_subtask_preserves_failure_status_with_diagnostics(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'private-test-key')
    def fail(req, **kwargs):
        raise urllib.error.HTTPError(req.full_url, 404, 'Not Found', {}, io.BytesIO(b'{"error":{"code":"ModelNotFound"}}'))
    monkeypatch.setattr(urllib.request, 'urlopen', fail)
    result = call_qwen_subtask_resolver({'frames': []}, base_url='https://example.test/v1/chat/completions')
    assert result['status'] == 'request_failed'
    assert 'ModelNotFound' in result['error']
    assert 'subtask_resolver' in result['error']


def test_redacts_echoed_credentials_and_url_credentials(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'private-test-key')
    def fail(req, **kwargs):
        raise urllib.error.HTTPError(req.full_url, 404, 'Not Found', {}, io.BytesIO(b'API key private-test-key is invalid'))
    monkeypatch.setattr(urllib.request, 'urlopen', fail)
    with pytest.raises(urllib.error.HTTPError) as caught:
        call_qwen_text_resolver(review_payload={'frames': []}, model='qwen-plus', base_url='https://user:password@example.test/v1/chat/completions?key=secret')
    message = str(caught.value)
    assert 'private-test-key' not in message
    assert 'password' not in message and 'key=secret' not in message
    assert 'invalid' in message


def test_redacts_response_echoed_url_and_reason_and_readable_body(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'private-test-key')
    url = 'https://user:password@example.test/v1/chat/completions?key=query-secret'
    reason = 'Not Found private-test-key query-secret password'
    body = f'failed URL {url}; token private-test-key'
    def fail(req, **kwargs):
        raise urllib.error.HTTPError(req.full_url, 404, reason, {}, io.BytesIO(body.encode()))
    monkeypatch.setattr(urllib.request, 'urlopen', fail)
    with pytest.raises(urllib.error.HTTPError) as caught:
        call_qwen_text_resolver(review_payload={'frames': []}, model='qwen-plus', base_url=url)
    messages = str(caught.value) + caught.value.read().decode()
    for secret in ('private-test-key', 'query-secret', 'password'):
        assert secret not in messages
    assert 'example.test/v1/chat/completions' in messages


def test_redacts_encoded_and_decoded_url_credentials(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'private-test-key')
    url = 'https://user:p%40ss@example.test/v1/chat/completions?key=secret%2Bvalue'
    def fail(req, **kwargs):
        raise urllib.error.HTTPError(req.full_url, 404, 'Not Found', {}, io.BytesIO(b'p@ss p%40ss secret+value secret%2Bvalue'))
    monkeypatch.setattr(urllib.request, 'urlopen', fail)
    with pytest.raises(urllib.error.HTTPError) as caught:
        call_qwen_text_resolver(review_payload={'frames': []}, model='qwen-plus', base_url=url)
    for secret in ('p@ss', 'p%40ss', 'secret+value', 'secret%2Bvalue'):
        assert secret not in str(caught.value)
