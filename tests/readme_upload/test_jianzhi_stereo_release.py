import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[2] / 'roboego_hand_vis/upload_data/'
          'readme_upload/guanglun/prepare_jianzhi_stereo_release.py')
spec = importlib.util.spec_from_file_location('stereo_release', SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'publish'
    sample = root / 'batch_0000001/sample'
    backup = tmp_path / 'backup'
    meta = sample / 'meta'
    meta.mkdir(parents=True)
    episode = sample / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    old = 'observation.images.head_left'
    features = {key: {'dtype': 'video'} for key in (*migration.KEEP, old)}
    features['observation.imu'] = {'dtype': 'float32'}
    info = dict(codebase_version='v3.1', total_episodes=1, total_videos=3,
                features=features, master_feature=old)
    original = json.dumps(info).encode()
    (meta / 'info.json').write_bytes(original)
    untouched = [meta / 'readme.json', meta / 'episodes_stats.jsonl',
                 episode / 'master_index.parquet', episode / 'observation.imu.parquet',
                 episode / (migration.KEEP[0] + '.parquet'),
                 sample / 'annotation/episode_000000_structured.json']
    # Retained media need not be probed, parsed or even present for this migration.
    for path in untouched:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'unchanged fixture')
    removed = [episode / (old + '.parquet'),
               sample / 'videos/chunk-000' / old / 'episode_000000.mp4']
    for path in removed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'old modality')
    return root, sample, backup, original, untouched, removed


def test_preview_and_apply_preserve_unrelated_files(dataset):
    root, sample, backup, original, untouched, removed = dataset
    assert migration.process(sample, root, backup, False)['status'] == 'ok'
    assert all(path.exists() for path in removed)
    assert not backup.exists()
    result = migration.process(sample, root, backup, True)
    assert result['status'] == 'ok', result
    saved = backup / sample.relative_to(root)
    assert (saved / 'meta/info.json').read_bytes() == original
    for path in removed:
        assert not path.exists()
        assert (saved / path.relative_to(sample)).read_bytes() == b'old modality'
    assert all(path.read_bytes() == b'unchanged fixture' for path in untouched)
    info = json.loads((sample / 'meta/info.json').read_bytes())
    assert info['codebase_version'] == 'v2.1'
    assert info['master_feature'] == migration.KEEP[0]
    assert set(info['features']) == {*migration.KEEP, 'observation.imu'}
    assert migration.plan(sample) == ({}, [])


def test_failed_info_write_restores_moved_modalities(dataset, monkeypatch):
    root, sample, backup, original, untouched, removed = dataset
    def fail(*args):
        raise OSError('simulated metadata write failure')
    monkeypatch.setattr(migration, 'atomic', fail)
    result = migration.process(sample, root, backup, True)
    assert result['status'] == 'error'
    assert (sample / 'meta/info.json').read_bytes() == original
    assert all(path.read_bytes() == b'old modality' for path in removed)
    assert all(path.read_bytes() == b'unchanged fixture' for path in untouched)


def test_existing_backup_cannot_be_overwritten(dataset):
    root, sample, backup, original, untouched, removed = dataset
    target = backup / sample.relative_to(root)
    target.mkdir(parents=True)
    result = migration.process(sample, root, backup, True)
    assert result['status'] == 'error'
    assert 'Backup already exists' in result['error']
    assert (sample / 'meta/info.json').read_bytes() == original
    assert all(path.exists() for path in removed)
