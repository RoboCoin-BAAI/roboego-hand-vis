from scripts.jianzhi import depth
from roboego_hand_vis.upload_data.ks3 import atomic_json


def test_discover_jianzhi_uploaded_and_cleaned_receipts(tmp_path):
    batch = tmp_path / 'batch_0000001'
    for name, status in [('ready', 'uploaded'), ('cleaned', 'complete'),
                         ('pending', 'annotated'), ('failed', 'uploaded')]:
        row = dict(status=status, destination=f'ks3://bucket/{name}/',
                   upload=dict(file_sizes={'meta/info.json': 2}))
        if name == 'failed':
            row['error'] = 'upload failed'
        atomic_json(batch / 'records' / f'{name}.json', row)
    assert {r['destination_uri'] for _, r in depth.discover(tmp_path, 1, 1)} == {
        'ks3://bucket/ready/', 'ks3://bucket/cleaned/'}
    atomic_json(batch / 'depth_done.json', ['ks3://bucket/ready/'])
    assert [r['destination_uri'] for _, r in depth.discover(tmp_path, 1, 1)] == [
        'ks3://bucket/cleaned/']
    assert list(depth.discover(tmp_path, 2, 3)) == []


def test_local_mode_forwards_depth_options(monkeypatch):
    from roboego_hand_vis.lerobot_fast_foundation_depth.L1 import run
    received = []
    monkeypatch.setattr(run, 'main', lambda args: received.append(args) or 0)
    args = ['--source', '/data/session', '--dry-run']
    assert depth.main(args) == 0
    assert received == [args + ['--hevc12-preset', 'fast']]


def test_receipt_preview_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(depth, 'Worker', lambda *a: (_ for _ in ()).throw(AssertionError()))
    assert depth.main(['--state-root', str(tmp_path), '--download-root', str(tmp_path / 'data'),
                       '--model-root', 'unused', '--ckpt', 'unused',
                       '--ks3-tool', 'unused', '--ks3-config', 'unused']) == 0
    assert not (tmp_path / 'data').exists()
