import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from roboego_hand_vis.lerobot_vio.L1.storage import stage_features


def test_camera_pose_includes_extrinsic_lever_arm(tmp_path):
    episode = tmp_path / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    timestamps = episode / 'left.parquet'
    pq.write_table(pa.table(dict(timestamp=[0., 1.], episode_index=[0, 0])), timestamps)
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps(dict(fps=30, features={})))
    imu = np.repeat(np.eye(4)[None], 2, axis=0)
    imu[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    extrinsic = np.eye(4)
    extrinsic[0, 3] = 1.
    camera = imu @ extrinsic
    staging = tmp_path / '.staging'
    staging.mkdir()
    metadata = stage_features(dict(session=tmp_path, episode=episode,
        left_timestamps=timestamps, publish_left_camera_pose=True),
        dict(status=np.array(['valid', 'valid']), T_world_imu=imu,
             T_world_camera=camera), staging, {})
    output = pq.read_table(staging / episode.relative_to(tmp_path) /
                           'observation.left_camera_pose.parquet').to_pydict()
    assert np.allclose(output['x'], [0., -1.])
    assert np.allclose(output['y'], [0., 1.])
    assert np.isclose(output['qz'][1], np.sqrt(.5))
    assert metadata['device_frame'] == 'rectified left camera'
    assert 'reference_T_world_camera' in metadata


def test_l3_success_compacts_only_selected_episode(tmp_path, monkeypatch):
    from roboego_hand_vis.lerobot_vio.L3 import compute
    episode = tmp_path / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    (episode / 'left_camera_pose_provenance.json').write_text('{}')
    output = tmp_path / 'vio_zupt/chunk-000/episode_000000'
    output.mkdir(parents=True)
    (output / 'report.json').write_text('{}')
    other = tmp_path / 'vio/chunk-000/episode_000001'
    other.mkdir(parents=True)
    monkeypatch.setattr(compute, 'shared_run_session', lambda *a: [
        dict(status='done', output=str(output), log='temporary')])
    result, = compute.run_session([dict(session=tmp_path, episode=episode)], {})
    assert not output.exists()
    assert other.exists()
    assert (tmp_path / 'annotation/vio/chunk-000/episode_000000/report.json').exists()
    assert 'log' not in result
