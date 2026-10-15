from copy import deepcopy
from pathlib import Path

import pytest


def report():
    return {
        'schema_version': 'roboego.vio.delivery.v2',
        'status': 'complete', 'execution_status': 'complete', 'returncode': 0,
        'quality_status': 'not_validated', 'pose_count': 2,
        'input': {'session': '/tmp/session', 'arguments': ['--input_mode', 'rectified']},
        'artifacts': {'trajectory': 'trajectory.npz'},
        'offline': {'status': 'completed', 'pose_count': 2},
        'frame_alignment': {'valid_pose_count': 2},
    }


def test_delivery_report_accepts_execution_without_claiming_quality():
    from roboego_hand_vis.vio.delivery import validate_report
    value = report()
    original = deepcopy(value)
    assert validate_report(value, session=Path('/tmp/session'), input_mode='rectified') == 2
    assert value == original


@pytest.mark.parametrize('change', [
    {'status': 'failed'}, {'execution_status': 'failed'}, {'returncode': 1},
    {'pose_count': 0}, {'pose_count': True},
    {'schema_version': 'unknown'},
    {'input': {'session': '/wrong', 'arguments': ['--input_mode', 'rectified']}},
    {'input': {'session': '/tmp/session', 'arguments': ['--input_mode', 'raw']}},
    {'artifacts': {'trajectory': '../outside.npz'}},
    {'offline': {'status': 'completed', 'pose_count': 3}},
    {'offline': {'status': 'failed', 'pose_count': 2}},
    {'frame_alignment': {'valid_pose_count': 3}},
])
def test_delivery_report_rejects_failed_mismatched_or_inconsistent_result(change):
    from roboego_hand_vis.vio.delivery import validate_report
    value = report()
    value.update(change)
    with pytest.raises(ValueError):
        validate_report(value, session=Path('/tmp/session'), input_mode='rectified')


def arrays():
    import numpy as np
    transforms = np.repeat(np.eye(4)[None], 3, axis=0)
    transforms[0] = np.nan
    return dict(schema_version=np.array('openvins.frame_aligned.v2'),
                quality_status=np.array('not_validated'),
                frame_index=np.arange(3, dtype=np.int64),
                right_frame_index=np.array([-1, 1, 2], dtype=np.int64),
                timestamp_camera=np.array([1., 2., 3.]), timestamp_imu=np.array([np.nan, 2., 3.]),
                status=np.array(['no_stereo_pair', 'valid', 'valid'], dtype='U32'),
                T_world_imu=transforms.copy(), T_world_camera=transforms.copy())


def test_frame_aligned_delivery_allows_invalid_rows(tmp_path):
    import json
    import numpy as np
    from roboego_hand_vis.vio.delivery import validate_delivery
    (tmp_path/'report.json').write_text(json.dumps(report()))
    np.savez(tmp_path/'trajectory.npz', **arrays())
    assert validate_delivery(tmp_path, session=Path('/tmp/session'), input_mode='rectified') == 2


@pytest.mark.parametrize('failure', ['nan_pose', 'rotation', 'shape', 'count', 'timestamps', 'index', 'schema', 'object'])
def test_frame_aligned_delivery_rejects_corruption(tmp_path, failure):
    import json
    import numpy as np
    from roboego_hand_vis.vio.delivery import validate_delivery
    a = arrays()
    if failure == 'nan_pose': a['T_world_imu'][1, 0, 3] = np.nan
    if failure == 'rotation': a['T_world_camera'][1, 0, 0] = 2
    if failure == 'shape': a['T_world_camera'] = a['T_world_camera'][:2]
    if failure == 'count': a['status'][1] = 'initializing'
    if failure == 'timestamps': a['timestamp_camera'][2] = 2
    if failure == 'index': a['frame_index'][2] = 1
    if failure == 'schema': a['schema_version'] = np.array('unknown')
    if failure == 'object': a['extra'] = np.array([{}], dtype=object)
    (tmp_path/'report.json').write_text(json.dumps(report()))
    np.savez(tmp_path/'trajectory.npz', **a)
    with pytest.raises(ValueError):
        validate_delivery(tmp_path, session=Path('/tmp/session'), input_mode='rectified')


def test_native_export_aligns_full_video_and_preserves_pose_indices(tmp_path):
    import json
    import numpy as np
    from roboego_hand_vis.vio.delivery import export_native, validate_delivery
    session = tmp_path/'session'
    (session/'timestamps').mkdir(parents=True)
    (session/'timestamps/left_timestamps.txt').write_text('0 1\n1 2\n2 3\n')
    work = tmp_path/'work'
    (work/'trajectory').mkdir(parents=True)
    pairs = work/'stereo_pairs.csv'
    pairs.write_text('timestamp,left_frame_index,right_frame_index\n2,1,5\n3,2,6\n')
    manifest = dict(session_dir=str(session), input_mode='rectified', stereo_pairs_path=str(pairs),
                    videos={'left': {'frame_count': 3}}, T_imu_camera0=np.eye(4).tolist())
    (work/'run_manifest.json').write_text(json.dumps(manifest))
    (work/'trajectory/run_report.json').write_text(json.dumps({'initialized': True, 'pose_count': 2,
        'artifacts': {'trajectory_jsonl': str(work/'trajectory/trajectory.jsonl')}}))
    rows = [dict(frame_index=i, right_frame_index=i+4, timestamp_camera=float(i+1), timestamp_imu=float(i+1),
                 orientation_world_imu_xyzw=[0,0,0,1], position_world_imu=[i,0,0]) for i in [1,2]]
    (work/'trajectory/trajectory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    out = tmp_path/'delivery'
    export_native(work, out)
    assert validate_delivery(out, session=session, input_mode='rectified') == 2
    assert 'artifacts' not in json.loads((out/'report.json').read_text())['offline']
    with np.load(out/'trajectory.npz') as a:
        assert a['status'].tolist() == ['no_stereo_pair', 'valid', 'valid']
        assert a['right_frame_index'].tolist() == [-1,5,6]
        assert a['T_world_camera'][2,0,3] == 2
        assert np.isnan(a['T_world_camera'][0]).all()
    # The native pipeline launches this entry as a file using its configured Python.
    import subprocess
    import sys
    from roboego_hand_vis.vio import export_entry
    cli_out = tmp_path/'cli-delivery'
    subprocess.run([sys.executable, export_entry.__file__, '--work', str(work), '--output', str(cli_out)],
                   cwd=tmp_path, check=True)
    assert validate_delivery(cli_out, session=session, input_mode='rectified') == 2
