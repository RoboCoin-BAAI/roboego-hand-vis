import csv
from roboego_hand_vis.lerobot_vio.L1.compute import configure_zupt
from roboego_hand_vis.vio.calibration import _estimator_config, _write_opencv_yaml, load_yaml_mapping


def test_toggle_only_zupt_flag(tmp_path):
    path = tmp_path / 'estimator.yaml'
    original = _estimator_config(tmp_path, downsample_cameras=False)
    _write_opencv_yaml(path, original)
    result = configure_zupt(path, True)
    configured = load_yaml_mapping(path)
    assert result['try_zupt'] is True
    assert configured == {**original, 'try_zupt': True}
    configure_zupt(path, False)
    assert load_yaml_mapping(path) == original


def test_zupt_separate_plan(tmp_path):
    from roboego_hand_vis.lerobot_vio.L1.compute import plan_session
    episode = tmp_path / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    (episode / 'observation.imu.parquet').touch()
    calibration = tmp_path / 'calibrations'
    calibration.mkdir()
    for name in ('camera_imu-camchain-imucam.yaml', 'camera_imu-imu.yaml'):
        (calibration / name).touch()
    for side in ('left', 'right'):
        key = f'observation.images.head_{side}'
        (episode / f'{key}.parquet').touch()
        folder = tmp_path / 'videos/chunk-000' / key
        folder.mkdir(parents=True)
        (folder / 'episode_000000.mp4').touch()
    assert plan_session(tmp_path, 'raw', zupt=True)[0]['output'] == tmp_path / 'vio_raw_zupt/chunk-000/episode_000000'
