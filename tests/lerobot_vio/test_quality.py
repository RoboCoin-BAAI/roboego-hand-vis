import numpy as np
import pytest

from roboego_hand_vis.lerobot_vio.L1.quality import check_trajectory
from roboego_hand_vis.lerobot_vio.L1.storage import quaternion


def trajectory(n=2):
    t = np.repeat(np.eye(4)[None], n, axis=0)
    return dict(schema_version=np.array('openvins.frame_aligned.v2'),
                frame_index=np.arange(n), right_frame_index=np.arange(n),
                timestamp_camera=np.arange(n, dtype=float), timestamp_imu=np.arange(n, dtype=float),
                status=np.full(n, 'valid', dtype='U32'), T_world_imu=t.copy(), T_world_camera=t.copy())


def test_boundary_missing_and_low_ratio():
    data = trajectory(5)
    data['status'][:3] = 'imu_unavailable'
    for key in ('T_world_imu', 'T_world_camera'):
        data[key][:3] = np.nan
    result = check_trajectory(data, np.arange(5))
    assert result['valid_ratio'] == .4
    assert result['first_valid_frame'] == 3
    assert result['reasons'] == ['low_valid_pose_ratio']


@pytest.mark.parametrize('mutation', ['rotation', 'homogeneous', 'finite', 'index', 'timestamp'])
def test_reject_corrupt_output(mutation):
    data = trajectory()
    if mutation == 'rotation':
        data['T_world_imu'][0, 0, 0] = -1
    elif mutation == 'homogeneous':
        data['T_world_camera'][0, 3, 3] = 0
    elif mutation == 'finite':
        data['T_world_imu'][0, 0, 3] = np.nan
    elif mutation == 'index':
        data['frame_index'][1] = 4
    else:
        data['timestamp_camera'][0] = .1
    with pytest.raises(ValueError):
        check_trajectory(data, [0., 1.])


def test_jump_speed_and_drift_heuristic():
    data = trajectory()
    data['T_world_imu'][1, 0, 3] = 60
    result = check_trajectory(data, [0., 1.])
    assert {'translation_jump', 'abnormal_speed', 'large_excursion_possible_drift'} <= set(result['reasons'])
    assert result['drift_accuracy'] == 'not_measurable_without_reference'


def test_gap_uses_elapsed_time():
    data = trajectory(3)
    data['status'][1] = 'initializing'
    for key in ('T_world_imu', 'T_world_camera'):
        data[key][1] = np.nan
    data['T_world_imu'][2, 0, 3] = 1
    result = check_trajectory(data, [0., 1., 2.])
    assert result['max_speed_m_s'] == .5


def test_hamilton_rotation_export():
    r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert np.allclose(quaternion(r), [0., 0., np.sqrt(.5), np.sqrt(.5)])
    assert np.allclose(quaternion(np.diag([1., -1., -1.])), [1., 0., 0., 0.])


def test_angular_speed_check():
    data = trajectory()
    data['T_world_imu'][1, :3, :3] = np.diag([-1., -1., 1.])
    result = check_trajectory(data, [0., 1.], {'max_angular_speed_deg_s': 100.})
    assert result['max_angular_speed_deg_s'] == 180.
    assert 'abnormal_angular_speed' in result['reasons']


def test_storage_rebase_missing_and_metadata(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from roboego_hand_vis.lerobot_vio.L1.storage import stage_features, FEATURE, STATUS_FEATURE
    episode = tmp_path / 'data/chunk-000/episode_000001'
    episode.mkdir(parents=True)
    timestamps = episode / 'left.parquet'
    pq.write_table(pa.table(dict(timestamp=[0., 1., 2.], episode_index=[1, 1, 1])), timestamps)
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps(dict(fps=30., features={'observation.imu': {'dtype': 'float32'}})))
    (tmp_path / 'meta/episodes_stats.jsonl').write_text(json.dumps(dict(episode_index=0, stats={'other': 1})) + '\n')
    data = trajectory(3)
    data['status'][0] = 'imu_unavailable'
    for key in ('T_world_imu', 'T_world_camera'):
        data[key][0] = np.nan
        data[key][1:, 0, 3] = [10., 11.]
        data[key][1:, :3, :3] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    staging = tmp_path / '.staging'
    staging.mkdir()
    stage_features(dict(session=tmp_path, episode=episode, left_timestamps=timestamps), data, staging, {})
    result = pq.read_table(staging / 'data/chunk-000/episode_000001' / f'{FEATURE}.parquet').to_pydict()
    assert np.isnan(result['x'][0])
    assert np.allclose(result['y'][1:], [0., -1.])
    assert np.allclose(result['qw'][1:], [1., 1.])
    info = json.loads((staging / 'meta/info.json').read_text())
    assert 'observation.imu' in info['features']
    assert info['features'][STATUS_FEATURE]['status_codes']['imu_unavailable'] == 2
    records = [json.loads(line) for line in (staging / 'meta/episodes_stats.jsonl').read_text().splitlines()]
    assert records[0]['stats'] == {'other': 1}
    assert records[1]['stats'][FEATURE]['count'] == [2]
