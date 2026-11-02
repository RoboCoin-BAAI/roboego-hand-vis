import json
from pathlib import Path
from unittest.mock import patch

import pytest

from roboego_hand_vis.download_data.guanglun.inventory import plan, write_plan, SOURCE, DESTINATION
from roboego_hand_vis.download_data.guanglun.run import cleanup, process, summarize, batch_complete
from roboego_hand_vis.upload_data.ks3 import atomic_json
from roboego_hand_vis.upload_data.ks3 import KS3


def test_download_accepts_opaque_etag_after_verified_transfer(tmp_path):
    client = KS3('unused', 'unused')
    target = tmp_path/'source.mcap'
    item = dict(source_uri=SOURCE+'x.mcap', size=4, etag='0'*32)
    target.write_bytes(b'old!')
    def transfer(source, destination):
        Path(destination).write_bytes(b'new!')
    with patch.object(client, 'copy', side_effect=transfer) as copy:
        client.download(item, target)
        copy.assert_called_once()
    assert target.read_bytes() == b'new!'


def test_download_rejects_wrong_size(tmp_path):
    client = KS3('unused', 'unused')
    item = dict(source_uri=SOURCE+'x.mcap', size=4, etag='0'*32)
    with patch.object(client, 'copy', side_effect=lambda s,d: Path(d).write_bytes(b'x')):
        with pytest.raises(RuntimeError, match='expected=4, actual=1'):
            client.download(item, tmp_path/'source.mcap')


def test_batches_are_stable(tmp_path):
    objects = [dict(uri=SOURCE+f'{i:04}.mcap', size=12, etag='multipart') for i in range(201)]
    value = plan(objects, SOURCE, DESTINATION)
    assert [len(v) for v in value['batches'].values()] == [100, 100, 1]
    write_plan(tmp_path, value)
    write_plan(tmp_path, value)
    with pytest.raises(ValueError, match='Inventory changed'):
        write_plan(tmp_path, plan(objects[:-1], SOURCE, DESTINATION))


@pytest.mark.parametrize('upload_fails', [False, True])
def test_upload_receipt_precedes_cleanup(tmp_path, upload_fails):
    item = plan([dict(uri=SOURCE+'session.mcap', size=12, etag='multipart')], SOURCE, DESTINATION)['batches']['batch_0000001'][0]
    batch = 'batch_0000001'
    shared, local = tmp_path/'share', tmp_path/'home'
    work = local/batch/item['id']
    (work/'converted/meta').mkdir(parents=True)
    (work/'source.mcap').write_bytes(b'original')
    (work/'converted/meta/info.json').write_text('{}')
    (work/'converted/meta/guanglun_source.json').write_text('{"rectification_status":"native_rectified"}')
    options = dict(state_root=shared, local_root=local, ks3_tool='unused', ks3_config='unused')
    def upload(*args):
        assert work.exists()
        if upload_fails:
            raise RuntimeError('size mismatch')
        return dict(verification='sizes matched')
    with patch('roboego_hand_vis.download_data.guanglun.run.KS3') as client:
        client.return_value.upload_directory.side_effect = upload
        process(item, batch, options)
    records = summarize(shared, batch, [item])
    index = json.loads((shared/batch/'rectification_done.json').read_text())
    assert work.exists() == upload_fails
    assert index == ([] if upload_fails else [item['destination_uri']])
    assert records[0]['status'] == ('failed' if upload_fails else 'uploaded')
    if not upload_fails:
        with patch('roboego_hand_vis.download_data.guanglun.run.KS3') as client:
            process(item, batch, options)
            client.assert_not_called()


def test_cleanup_rejects_root(tmp_path):
    with pytest.raises(ValueError, match='Unsafe cleanup'):
        cleanup(tmp_path, tmp_path)


@pytest.mark.parametrize('status', ['rejected', 'conversion_failed', 'failed'])
def test_quality_failure_is_terminal_and_does_not_block(tmp_path, status):
    item = dict(id='abc', destination_uri=DESTINATION+'session/')
    shared, local = tmp_path/'share', tmp_path/'home'
    work = local/'batch_0000001'/'abc'
    work.mkdir(parents=True)
    (work/'source.mcap').write_bytes(b'raw')
    error = 'ValueError: Invalid semantic interval: 491.0, 491.0' if status == 'failed' else 'pose jump'
    atomic_json(shared/'batch_0000001/records/abc.json', dict(item, status=status, error=error))
    with patch('roboego_hand_vis.download_data.guanglun.run.KS3') as client:
        process(item, 'batch_0000001', dict(state_root=shared, local_root=local))
        client.assert_not_called()
    records = summarize(shared, 'batch_0000001', [item])
    assert records[0]['status'] == 'conversion_failed'
    assert records[0]['error'] == error
    assert not work.exists()
    assert batch_complete(records)
    assert json.loads((shared/'batch_0000001/rectification_done.json').read_text()) == []
    assert not batch_complete([dict(status='failed', local_cleaned=True)])
    assert not batch_complete([dict(status='conversion_failed', local_cleaned=False)])


@pytest.mark.parametrize('exit_code', [1, 2])
def test_converter_failure_does_not_block_batch(tmp_path, exit_code):
    item = dict(id='abc', destination_uri=DESTINATION+'session/')
    shared, local = tmp_path/'share', tmp_path/'home'
    options = dict(state_root=shared, local_root=local, ks3_tool='unused', ks3_config='unused',
                   python='unused', converter_repo=tmp_path, ffmpeg='unused')
    def fail(*args, **kwargs):
        kwargs['stdout'].write('ValueError: Invalid semantic interval: 491.0, 491.0')
        from types import SimpleNamespace
        return SimpleNamespace(returncode=exit_code)
    with patch('roboego_hand_vis.download_data.guanglun.run.KS3') as client, patch(
            'roboego_hand_vis.download_data.guanglun.run.subprocess.run', side_effect=fail):
        process(item, 'batch_0000001', options)
        client.return_value.upload_directory.assert_not_called()
    rows = summarize(shared, 'batch_0000001', [item])
    assert batch_complete(rows)
    assert rows[0]['status'] == 'conversion_failed'
    assert rows[0]['converter_returncode'] == exit_code
    assert not (local/'batch_0000001/abc').exists()


@pytest.mark.parametrize('metadata', [None, '{}', '{broken', '[]'])
def test_invalid_conversion_output_does_not_block(tmp_path, metadata):
    item = dict(id='abc', destination_uri=DESTINATION+'session/')
    shared, local = tmp_path/'share', tmp_path/'home'
    work = local/'batch_0000001/abc'
    (work/'converted/meta').mkdir(parents=True)
    (work/'converted/meta/info.json').write_text('{}')
    if metadata is not None:
        (work/'converted/meta/guanglun_source.json').write_text(metadata)
    with patch('roboego_hand_vis.download_data.guanglun.run.KS3') as client:
        process(item, 'batch_0000001', dict(state_root=shared, local_root=local,
                ks3_tool='unused', ks3_config='unused'))
        client.return_value.upload_directory.assert_not_called()
    rows = summarize(shared, 'batch_0000001', [item])
    assert batch_complete(rows)
    assert rows[0]['failure_reason'] == 'invalid_conversion_output'
    assert not work.exists()
    assert json.loads((shared/'batch_0000001/rectification_done.json').read_text()) == []
