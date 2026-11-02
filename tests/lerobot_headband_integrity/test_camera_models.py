import copy
import json
from types import SimpleNamespace

import pytest
import yaml

from roboego_hand_vis.lerobot_headband_integrity.camera_calibration import camera_error
from roboego_hand_vis.lerobot_headband_integrity.L1.calibration_repair import complete_vio_file
from roboego_hand_vis.lerobot_headband_integrity.L3.compute import check_calibration


def ds():
    return dict(camera_model='ds', distortion_model='none', distortion_coeffs=[],
                intrinsics=[-.19, .62, 594., 660., 955., 598.], resolution=[1920, 1200],
                timeshift_cam_imu=-.043,
                T_cn_cnm1=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                T_cam_imu=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


@pytest.mark.parametrize('field,value,message', [
    ('intrinsics', [-.19, .62, 594., 660.], 'six'),
    ('intrinsics', [-.19, 1.2, 594., 660., 955., 598.], 'alpha'),
    ('intrinsics', [-.19, .62, -594., 660., 955., 598.], 'positive'),
    ('intrinsics', [-.19, .62, 594., 660., float('nan'), 598.], 'finite'),
    ('distortion_coeffs', [0, 0, 0, 0], 'distortion'),
    ('resolution', [1920, 1200.5], 'integer'),
    ('camera_model', 'unknown', 'Unsupported'),
])
def test_invalid_model_parameters(field, value, message):
    camera = ds()
    camera[field] = value
    assert message in camera_error(camera)


def test_ds_l1_and_l3_accept_negative_xi_empty_distortion(tmp_path):
    assert camera_error(ds()) is None
    chain = {f'cam{i}': ds() for i in range(4)}
    path = tmp_path / 'camera_imu-camchain-imucam.yaml'
    path.write_text(yaml.safe_dump(chain))
    assert complete_vio_file(path) == (True, None)
    episode = tmp_path / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    (tmp_path / 'meta').mkdir()
    head = {'camera-camchain': chain, 'camera_imu-camchain-imucam': copy.deepcopy(chain),
            'camera_imu-imu': {'imu0': dict(update_rate=200, accelerometer_noise_density=.01, accelerometer_random_walk=.001, gyroscope_noise_density=.01, gyroscope_random_walk=.001)}}
    (tmp_path / 'meta/calibrations.jsonl').write_text(json.dumps(dict(
        episode_index=0, calibrations={'ego_headband': head})))
    checks = check_calibration(SimpleNamespace(root=tmp_path, episodes=[episode]))
    assert all(c['passed'] for c in checks), checks


def test_pinhole_still_validates_distortion_and_focal_length():
    camera = dict(camera_model='pinhole', distortion_model='equidistant',
                  intrinsics=[500, 500, 320, 240], resolution=[640, 480], distortion_coeffs=[0]*4)
    assert camera_error(camera) is None
    camera['distortion_coeffs'] = []
    assert camera_error(camera)
    camera['distortion_coeffs'] = [0]*4
    camera['intrinsics'][0] = -1
    assert camera_error(camera)


@pytest.mark.parametrize('model', ['ds', 'pinhole'])
@pytest.mark.parametrize('damage', ['none', 'zero_rotation', 'reflection', 'missing_transform', 'missing_timeshift', 'empty_imu', 'negative_noise'])
def test_l3_strict_calibration(tmp_path, model, damage):
    camera = ds()
    if model == 'pinhole':
        camera.update(camera_model='pinhole', distortion_model='equidistant',
                      intrinsics=[500, 500, 320, 240], distortion_coeffs=[0.] * 4)
    chain = {f'cam{i}': copy.deepcopy(camera) for i in range(4)}
    imu = dict(update_rate=200, accelerometer_noise_density=.01,
               accelerometer_random_walk=.001, gyroscope_noise_density=.01,
               gyroscope_random_walk=.001)
    head = {'camera-camchain': copy.deepcopy(chain),
            'camera_imu-camchain-imucam': chain, 'camera_imu-imu': {'imu0': imu}}
    cam = chain['cam1']
    if damage == 'zero_rotation':
        cam['T_cam_imu'] = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]
    elif damage == 'reflection':
        cam['T_cam_imu'][0][0] = -1
    elif damage == 'missing_transform':
        del cam['T_cn_cnm1']
    elif damage == 'missing_timeshift':
        del cam['timeshift_cam_imu']
    elif damage == 'empty_imu':
        imu.clear()
    elif damage == 'negative_noise':
        imu['gyroscope_noise_density'] = -1
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/calibrations.jsonl').write_text(json.dumps(dict(
        episode_index=0, calibrations={'ego_headband': head})))
    ctx = SimpleNamespace(root=tmp_path, episodes=[tmp_path / 'episode_000000'])
    checks = check_calibration(ctx)
    assert all(c['passed'] for c in checks) is (damage == 'none')
