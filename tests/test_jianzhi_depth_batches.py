from types import SimpleNamespace
from scripts.jianzhi import depth_batches as runner
from scripts.L1.gpu_supplement_worker import options
import pytest


def test_remote_discovery_is_lazy_and_ignores_root(tmp_path):
    calls = []
    class Client:
        def _run(self, command):
            parent = command[-1]
            calls.append(parent)
            if parent == 'ks3://bucket/root/':
                return parent+'\n'+parent+'batch_0000001/\n'+parent+'batch_0000002/\n'
            return parent+'\n'+parent+'session/\n'
    from unittest.mock import patch
    a = SimpleNamespace(runtime_dir=tmp_path, ks3_tool=tmp_path/'tool',
                        ks3_config=tmp_path/'config', remote_root='ks3://bucket/root/',
                        first_batch=1, last_batch=2)
    with patch.object(runner.download_depth, 'Client', return_value=Client()):
        iterator = runner.discover_batches(a)
        first = next(iterator)
        assert runner.read_batch(first) == ['ks3://bucket/root/batch_0000001/session/']
        assert calls == ['ks3://bucket/root/batch_0000001/']
        assert next(iterator).stem == 'batch_0000002'


def test_missing_batch_does_not_hide_later_batch(tmp_path, monkeypatch):
    calls = []
    class Client:
        def _run(self, command):
            parent = command[-1]
            calls.append(parent)
            if parent.endswith('batch_0000022/'):
                return parent+'\n'
            return parent+'session/\n'
    a = SimpleNamespace(runtime_dir=tmp_path, ks3_tool=tmp_path/'tool',
                        ks3_config=tmp_path/'config', remote_root='ks3://bucket/root/',
                        first_batch=22, last_batch=23)
    monkeypatch.setattr(runner.download_depth, 'Client', lambda *args: Client())
    assert [p.stem for p in runner.discover_batches(a)] == ['batch_0000023']
    assert len(calls) == 2
    assert (tmp_path/'failed/batch_0000022/discovery.json').exists()


@pytest.mark.parametrize('fail_first', [False, True])
def test_batches_download_then_compute_keep_one_worker(tmp_path, monkeypatch, fail_first):
    batches = []
    for number in (1, 2):
        csv = tmp_path/f'batch_{number:07d}.csv'
        csv.write_text('目标路径\nks3://bucket/session/\n', encoding='utf-8')
        batches.append(csv)
    events = []
    monkeypatch.setattr(runner.download_depth, 'main', lambda args: events.append('download') or 0)
    monkeypatch.setattr(runner, 'plan_session', lambda s: [dict(episode=str(s))])
    monkeypatch.setattr(runner, 'existing_valid', lambda *a, **k: True)
    class Worker:
        def __init__(self, *args):
            events.append('load')
        def execute(self, cmd, log):
            assert options(cmd).valid_iters == 4
            assert options(cmd).hevc12_preset == 'fast'
            events.append('compute')
            return int(fail_first and 'batch_0000001' in str(log))
        def close(self):
            events.append('close')
    monkeypatch.setattr(runner, 'Worker', Worker)
    a = SimpleNamespace(devices=['0'], runtime_dir=tmp_path/'runtime', source=tmp_path/'data',
                        model_root=tmp_path/'model', ckpt=tmp_path/'ckpt', jobs=64,
                        batch_size=4, valid_iters=4, hevc12_preset='fast', ks3_tool='tool', ks3_config='config',
                        remote_root='ks3://bucket/processed/', ffmpeg='ffmpeg', dry_run=False)
    assert runner.run_batches(a, batches) == int(fail_first)
    assert events == ['download', 'load', 'compute', 'download', 'compute', 'close']
    events.clear()
    assert runner.run_batches(a, batches) == int(fail_first)
    if fail_first:
        assert events == ['download', 'load', 'compute', 'download', 'close']
        assert (a.runtime_dir/'failed/batch_0000001/session.json').exists()
    else:
        assert events == ['download', 'download']
