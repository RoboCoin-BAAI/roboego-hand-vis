import pytest

from roboego_hand_vis.lerobot_vio.L1.compute import configure_initialization
from roboego_hand_vis.vio.calibration import _estimator_config, _write_opencv_yaml, load_yaml_mapping


@pytest.mark.parametrize('zupt', [False, True])
def test_initialization_changes_only_dynamic_flag(tmp_path, zupt):
    config = tmp_path / 'estimator.yaml'
    original = _estimator_config(tmp_path, downsample_cameras=False)
    original['try_zupt'] = zupt
    _write_opencv_yaml(config, original)
    settings = configure_initialization(config, 'static')
    assert load_yaml_mapping(config) == {**original, 'init_dyn_use': False}
    assert settings['wait_for_jerk'] is (not zupt)
    configure_initialization(config, 'auto')
    assert load_yaml_mapping(config) == original
    before = config.read_bytes()
    with pytest.raises(ValueError, match='init_mode'):
        configure_initialization(config, 'invalid')
    assert config.read_bytes() == before


def test_dynamic_disables_static_initializer(tmp_path):
    config = tmp_path / 'estimator.yaml'
    original = _estimator_config(tmp_path, downsample_cameras=False)
    _write_opencv_yaml(config, original)
    settings = configure_initialization(config, 'dynamic')
    assert settings['init_dyn_use'] is True
    assert settings['init_imu_thresh'] == 0.0
    assert load_yaml_mapping(config) == {**original, 'init_dyn_use': True, 'init_imu_thresh': 0.0}
