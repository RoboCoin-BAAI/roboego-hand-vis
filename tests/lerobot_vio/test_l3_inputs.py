"""CPU integration of L3 JSONL -> rectification -> depth/VIO loaders."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import Encoder
from roboego_hand_vis.lerobot_stereo_rectification.L3.compute import process_session
from roboego_hand_vis.lerobot_fast_foundation_depth.L3.compute import episode_inputs
from roboego_hand_vis.lerobot_vio.L3.compute import plan_session, materialize_calibration
from roboego_hand_vis.lerobot_vio.L1.compute import prepare


def fixture(root):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    episode = root / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    (root / 'meta').mkdir()
    features, chain = {}, {}
    transform = np.eye(4)
    transform[0, 3] = -.1
    for i, side in enumerate(('left', 'right')):
        key = f'observation.images.headband_{side}_rgb'
        features[key] = dict(dtype='video', shape=[48,64,3], fps=30)
        pq.write_table(pa.table({'timestamp': [0., 1/30, 2/30], 'episode_index': [0,0,0]}), episode / f'{key}.parquet')
        encoder = Encoder(root / 'videos/chunk-000' / key / 'episode_000000.mp4', (64,48), 30, ffmpeg)
        try:
            for frame in range(3):
                encoder.write(np.full((48,64,3), 60+frame*20, dtype=np.uint8))
        finally:
            encoder.close()
        chain[f'cam{i}'] = dict(camera_model='pinhole', distortion_model='equidistant',
             intrinsics=[32.,32.,32.,24.], distortion_coeffs=[0.,0.,0.,0.], resolution=[64,48],
             rostopic=f'/fisheye/{side}/image_raw/compressed', T_cam_imu=(np.eye(4) if i==0 else transform).tolist(),
             timeshift_cam_imu=0.)
    chain['cam1']['T_cn_cnm1'] = transform.tolist()
    imu = dict(imu0=dict(update_rate=200, time_offset=0.,
               accelerometer_noise_density=.01, accelerometer_random_walk=.001,
               gyroscope_noise_density=.001, gyroscope_random_walk=.0001))
    head = {'camera-camchain': chain, 'camera_imu-camchain-imucam': chain, 'camera_imu-imu': imu}
    (root / 'meta/calibrations.jsonl').write_text(json.dumps(dict(episode_index=0, calibrations={'ego_headband':head})))
    (root / 'meta/info.json').write_text(json.dumps(dict(fps=30, features=features)))
    samples = 25
    table = dict(timestamp=[-.02+i*.005 for i in range(samples)],
                 **{k: [9.81 if k=='az' else 0.]*samples for k in ('ax','ay','az','gx','gy','gz')})
    pq.write_table(pa.table(table), episode / 'observation.headband_imu0.parquet')
    return episode, ffmpeg


def test_l3_loaders_and_prepare(tmp_path):
    episode, ffmpeg = fixture(tmp_path)
    source = (tmp_path / 'meta/calibrations.jsonl').read_bytes()
    raw, = plan_session(tmp_path, input_mode='raw')
    assert raw['imu'].name == 'observation.headband_imu0.parquet'
    assert not (tmp_path / 'calibrations').exists()
    raw_inputs = prepare(raw, tmp_path / 'raw_work')
    assert raw_inputs['imu'].is_file()
    process_session(tmp_path, dict(ffmpeg=ffmpeg))
    depth = episode_inputs(tmp_path, episode)
    assert 'headband_left_rgb_rectified' in depth['manifest']['videos']['left']['path']
    assert depth['timestamps'].stem == 'observation.images.headband_left_rgb_rectified'
    plan, = plan_session(tmp_path)
    from unittest.mock import patch
    with patch('roboego_hand_vis.common.stereo_quality.require_stereo_quality', side_effect=AssertionError('Runtime first-frame quality gate must not run')):
        prepared = prepare(plan, tmp_path / 'work')
    assert prepared['config'].is_file()
    assert prepared['stereo_pairs'].read_text().count('\n') == 4
    assert (tmp_path / 'work/source_calibration/source.json').is_file()
    assert (tmp_path / 'meta/calibrations.jsonl').read_bytes() == source
    assert not (tmp_path / 'calibrations/camera_imu-imu.yaml').exists()
    plan['imu'].unlink()
    with pytest.raises(ValueError, match='Missing imu'):
        plan_session(tmp_path)


def test_missing_episode_calibration_fails(tmp_path):
    fixture(tmp_path)
    path = tmp_path / 'meta/calibrations.jsonl'
    row = json.loads(path.read_text())
    row['episode_index'] = 9
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Expected one calibration'):
        plan_session(tmp_path, input_mode='raw')
