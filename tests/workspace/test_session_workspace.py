import json
from types import SimpleNamespace

import pytest


def workspace(tmp_path, stage='raw_headband_integrity'):
    from roboego_hand_vis.common.session_workspace import SessionWorkspace
    shared, local = tmp_path / 'shared', tmp_path / 'local'
    (shared / 'session/videos').mkdir(parents=True, exist_ok=True)
    (shared / 'session/videos/left.mkv').write_bytes(b'original')
    return SessionWorkspace(shared, local, stage, copy_missing=stage == 'raw_headband_integrity')


def test_first_stage_copies_missing_session_and_reuses_existing_directory(tmp_path):
    ws = workspace(tmp_path)
    local = ws.prepare('session')
    assert (local / 'videos/left.mkv').read_bytes() == b'original'
    (local / 'videos/left.mkv').write_bytes(b'existing local file')
    ws.release()
    assert ws.prepare('session') == local
    assert (local / 'videos/left.mkv').read_bytes() == b'existing local file'


def test_later_stage_requires_local_session_but_accepts_manually_prepared_directory(tmp_path):
    ws = workspace(tmp_path, 'vio')
    with pytest.raises(ValueError, match='Local session is missing'):
        ws.prepare('session')
    (ws.local_root / 'session').mkdir()
    assert ws.prepare('session') == ws.local_root / 'session'


def test_copy_outputs_keeps_json_bytes_unchanged_and_excludes_raw_inputs(tmp_path):
    ws = workspace(tmp_path)
    local = ws.prepare('session')
    report = local / 'annotation/quality_report.json'
    report.parent.mkdir()
    original = json.dumps({'session': str(local), 'gate': {'passed': True}}, indent=4).encode()
    report.write_bytes(original)
    (local / 'videos/left.mkv').write_bytes(b'changed local video')
    (local / 'annotation/unrelated.json').write_text('{}')
    assert 'annotation/quality_report.json' in ws.publish('completed')
    assert (ws.shared_root / 'session/annotation/quality_report.json').read_bytes() == original
    assert (ws.shared_root / 'session/videos/left.mkv').read_bytes() == b'original'
    assert not (ws.shared_root / 'session/annotation/unrelated.json').exists()


def test_copy_input_also_keeps_manifest_unchanged(tmp_path):
    ws = workspace(tmp_path)
    manifest = ws.shared_root / 'session/manifest.json'
    original = json.dumps({'path': str(ws.shared_root / 'session/videos/left.mkv')}).encode()
    manifest.write_bytes(original)
    local = ws.prepare('session')
    assert (local / 'manifest.json').read_bytes() == original


def test_interrupted_initial_copy_does_not_create_active_local_session(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr('shutil.copytree', fail)
    with pytest.raises(OSError, match='disk full'):
        ws.prepare('session')
    assert not (ws.local_root / 'session').exists()


class Service:
    database = SimpleNamespace(close=lambda: None)
    def __init__(self):
        self.finished, self.row = [], {'id': 1, 'local_path': 'session'}
    def claim_raw_headband_integrity(self):
        row, self.row = self.row, None
        return row
    def finish_raw_headband_integrity(self, data_id, *, passed):
        self.finished.append(passed)
        return True


def test_block_copies_report_before_database_update(tmp_path):
    from roboego_hand_vis.database.workspace import WorkspaceService
    ws, service = workspace(tmp_path), Service()
    wrapped = WorkspaceService(service, ws)
    row = wrapped.claim_raw_headband_integrity()
    report = ws.local_root / 'session/annotation/quality_report.json'
    report.parent.mkdir()
    report.write_text('{"gate": {"passed": false}}')
    original = service.finish_raw_headband_integrity
    def finish(data_id, *, passed):
        assert (ws.shared_root / 'session/annotation/quality_report.json').is_file()
        return original(data_id, passed=passed)
    service.finish_raw_headband_integrity = finish
    assert wrapped.finish_raw_headband_integrity(row['id'], passed=False)
    assert service.finished == [False]


def test_copy_failure_writes_failed_and_keeps_local_outputs(tmp_path, monkeypatch):
    from roboego_hand_vis.database.workspace import WorkspaceService
    ws, service = workspace(tmp_path), Service()
    wrapped = WorkspaceService(service, ws)
    row = wrapped.claim_raw_headband_integrity()
    report = ws.local_root / 'session/annotation/quality_report.json'
    report.parent.mkdir()
    report.write_text('{}')
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(ws, '_copy_file', fail)
    with pytest.raises(RuntimeError, match='disk full'):
        wrapped.finish_raw_headband_integrity(row['id'], passed=True)
    assert service.finished == [None]
    assert report.is_file()
    assert ws.active_session is None


def test_only_required_is_copied_paths_is_output_catalog(tmp_path):
    ws = workspace(tmp_path)
    local = ws.prepare('session')
    report = local / 'annotation/quality_report.json'
    report.parent.mkdir()
    report.write_text('{}')
    extra = local / 'annotation/extra.json'
    extra.write_text('{}')
    ws.rules = {'paths': ['annotation'], 'required': ['annotation/quality_report.json']}
    assert ws.publish('completed') == ['annotation/quality_report.json']
    assert not (ws.shared_root / 'session/annotation/extra.json').exists()


def test_required_supports_directory_and_glob(tmp_path):
    ws = workspace(tmp_path)
    local = ws.prepare('session')
    (local / 'result/subdir').mkdir(parents=True)
    (local / 'result/subdir/keep.json').write_text('{}')
    (local / 'videos/processed.mp4').write_bytes(b'processed')
    ws.rules = {'paths': [], 'required': ['result', 'videos/*.mp4']}
    assert ws.publish('completed') == ['result/subdir/keep.json', 'videos/processed.mp4']


def test_empty_required_means_no_output_copy(tmp_path):
    ws = workspace(tmp_path)
    ws.prepare('session')
    ws.rules = {'paths': ['videos'], 'required': []}
    assert ws.publish('completed') == []
