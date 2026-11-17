import pytest
from roboego_hand_vis.upload_data.ks3 import KS3


def test_recursive_upload_matches_download_cp_and_verifies_sizes(tmp_path, monkeypatch):
    (tmp_path/'meta').mkdir()
    (tmp_path/'meta/info.json').write_text('{}')
    client = KS3('ks3', 'config')
    commands = []
    monkeypatch.setattr(client, '_run', lambda args: commands.append(args))
    destination = 'ks3://bucket/dataset/session/'
    monkeypatch.setattr(client, 'objects', lambda uri: [dict(uri=uri+'meta/info.json', size=2)])
    result = client.upload_directory(tmp_path, destination, recursive_copy=True)
    assert len(commands) == 1
    command = commands[0]
    assert command == ['-r','-u','--read-timeout','6','--jobs=64','cp',str(tmp_path.resolve())+'/',destination]
    assert result['file_sizes'] == {'meta/info.json': 2}


def test_recursive_upload_still_rejects_remote_mismatch(tmp_path, monkeypatch):
    (tmp_path/'meta').mkdir()
    (tmp_path/'meta/info.json').write_text('{}')
    client = KS3('ks3','config')
    monkeypatch.setattr(client, '_run', lambda args: '')
    monkeypatch.setattr(client, 'objects', lambda uri: [])
    with pytest.raises(RuntimeError, match='verification failed'):
        client.upload_directory(tmp_path,'ks3://bucket/dataset/session/',recursive_copy=True)
