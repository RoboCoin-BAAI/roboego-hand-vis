import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.guanglun import depth
from scripts.L1.gpu_supplement_worker import options as worker_options
from roboego_hand_vis.upload_data.ks3 import atomic_json


def test_resident_depth_worker_accepts_batch_size():
    args = worker_options(['worker.py', 'depth', '--source', '/tmp/session',
        '--model-root', '/tmp/model', '--batch-size', '6'])
    assert args.batch_size == 6


def prepare(root, number):
    batch = root/f'batch_{number:07d}'
    uri = f'ks3://bucket/converted/{batch.name}/session/'
    row = dict(status='uploaded', destination_uri=uri,
               upload=dict(file_sizes={'meta/info.json': 2}))
    atomic_json(batch/'rectification_done.json', [uri])
    atomic_json(batch/'records/session.json', row)
    return batch, uri


def test_discover_reads_ready_receipts_and_skips_completed(tmp_path):
    b1, u1 = prepare(tmp_path, 1)
    b2, u2 = prepare(tmp_path, 2)
    atomic_json(b1/'depth_done.json', [u1])
    assert [r['destination_uri'] for b,r in depth.discover(tmp_path, 1, 574)] == [u2]
    assert list(depth.discover(tmp_path, 1, 1)) == []


def test_download_blocks_path_traversal(tmp_path):
    client = Mock()
    row = dict(destination_uri='ks3://bucket/session/',
               upload=dict(file_sizes={'meta/info.json': 2, '../escape': 3}))
    with pytest.raises(ValueError, match='Unsafe'):
        depth.download_dataset(client, row, tmp_path/'session')


@pytest.mark.parametrize('returncode', [0, 1])
def test_resident_spans_batches_and_marks_only_success(tmp_path, monkeypatch, returncode):
    root, output, model = tmp_path/'state', tmp_path/'download', tmp_path/'model'
    model.mkdir()
    checkpoint = model/'weights.pth'
    checkpoint.write_bytes(b'weights')
    (model/'cfg.yaml').write_text('{}')
    batches = [prepare(root, n) for n in (1, 2)]
    created = []
    class Worker:
        def __init__(self, *args):
            self.commands = []
            self.process = Mock()
            self.process.poll.return_value = None
            created.append(self)
        def execute(self, command, log):
            self.commands.append(command)
            return returncode
        def close(self):
            self.closed = True
    def download(client, row, target):
        target.mkdir(parents=True)
    monkeypatch.setattr(depth, 'Worker', Worker)
    monkeypatch.setattr(depth, 'download_dataset', download)
    monkeypatch.setattr(depth.shutil, 'which', lambda p: p)
    from roboego_hand_vis.lerobot_fast_foundation_depth.L1 import compute, reuse
    monkeypatch.setattr(compute, 'plan_session', lambda s: [dict(episode=str(s))])
    monkeypatch.setattr(reuse, 'existing_valid', lambda p, **kw: True)
    result = depth.main(['--state-root', str(root), '--download-root', str(output),
        '--model-root', str(model), '--ckpt', str(checkpoint), '--ks3-tool', 'unused',
        '--ks3-config', 'unused', '--min-free-gb', '0', '--batch-size', '3', '--once', '--execute'])
    assert result == returncode
    assert len(created) == 1 and len(created[0].commands) == 2 and created[0].closed
    assert all(command[command.index('--batch-size') + 1] == '3' for command in created[0].commands)
    for batch, uri in batches:
        assert depth.load(batch/'depth_done.json', []) == ([] if returncode else [uri])
    assert not (root/'depth_runner.lock').exists()


def test_preview_has_no_download_or_model_load(tmp_path, monkeypatch):
    root = tmp_path/'state'
    prepare(root, 1)
    worker = Mock()
    monkeypatch.setattr(depth, 'Worker', worker)
    assert depth.main(['--state-root', str(root), '--download-root', str(tmp_path/'data'),
        '--model-root', 'unused', '--ckpt', 'unused', '--ks3-tool', 'unused', '--ks3-config', 'unused']) == 0
    worker.assert_not_called()
    assert not (tmp_path/'data').exists()
