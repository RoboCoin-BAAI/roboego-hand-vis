from copy import deepcopy

import numpy as np
import pytest

from roboego_hand_vis.lerobot_stereo_rectification.jianzhi.compute import geometry_from_jianzhi


def calibration():
    # Common body rotation makes a reversed transform or subtraction of body
    # translations fail: the baseline must be expressed in the right camera.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    data = {}
    for name, x in [('head_left', 0.), ('head_right', .06)]:
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = rotation @ [x, 0., 0.] + [1., 2., 3.]
        data[name] = dict(camera_model='pinhole', distortion_model='ds',
            intrinsics=[500., 501., 800., 650.],
            distortion_coeffs=[500., 501., 800., 650., -.003, .57],
            resolution=[1600, 1300], T_b_c_matrix=pose.tolist())
    return data


def test_ds_order_and_body_pose_direction():
    data = calibration()
    original = deepcopy(data)
    left, right, transform = geometry_from_jianzhi(data)
    assert left['intrinsics'] == [-.003, .57, 500., 501., 800., 650.]
    assert right['camera_model'] == 'ds'
    np.testing.assert_allclose(transform[:3, :3], np.eye(3))
    np.testing.assert_allclose(transform[:3, 3], [-.06, 0., 0.], atol=1e-12)
    assert data == original


@pytest.mark.parametrize('problem', ['reflection', 'zero_baseline', 'wrong_ds_order'])
def test_invalid_calibration_rejected(problem):
    data = calibration()
    if problem == 'reflection':
        data['head_right']['T_b_c_matrix'][2][2] = -1.
    elif problem == 'zero_baseline':
        data['head_right']['T_b_c_matrix'] = data['head_left']['T_b_c_matrix']
    else:
        data['head_right']['distortion_coeffs'] = [.57, -.003, 500., 501., 800., 650.]
    with pytest.raises(ValueError):
        geometry_from_jianzhi(data)


@pytest.mark.parametrize('dry_run,json_exists', [(False, True), (False, False), (True, True)])
def test_yaml_selected_and_json_removed_only_for_real_run(tmp_path, monkeypatch, dry_run, json_exists):
    import json
    import yaml
    from roboego_hand_vis.lerobot_stereo_rectification.jianzhi import compute

    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps({'features': {
        f'observation.images.{side}': {'dtype': 'video'}
        for side in ('head_left', 'head_right')}}))
    episode = tmp_path / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    for side in ('head_left', 'head_right'):
        (episode / f'observation.images.{side}.parquet').touch()
    cal = tmp_path / 'calibrations'
    cal.mkdir()
    (cal / 'camera-camchain.yaml').write_text(yaml.safe_dump(calibration()))
    legacy = cal / 'jianzhi-camera-calibration.json'
    if json_exists:
        legacy.write_text('invalid JSON must never be read')
    quality_report = tmp_path / 'meta/jianzhi_quality_report.json'
    if json_exists:
        quality_report.write_text('{}')

    def rectify(*args, **kwargs):
        assert kwargs['write_frame_mapping'] is False
        assert kwargs['full_decode_validation'] is False
        recorded = yaml.safe_load((cal / 'camera-camchain.yaml').read_text())
        assert ('head_local_to_calibration_body' in recorded) == (not dry_run)
        assert legacy.exists() == (dry_run and json_exists)
        assert quality_report.exists() == (dry_run and json_exists)
        assert kwargs['calibration_source']['path'] == 'calibrations/camera-camchain.yaml'
        return 'processed'

    monkeypatch.setattr(compute, 'rectify_episode', rectify)
    assert compute.process_session(tmp_path, {'dry_run': dry_run}) == ['processed']


def test_frame_relation_preserves_cameras_and_is_idempotent(tmp_path):
    import yaml
    from roboego_hand_vis.lerobot_stereo_rectification.jianzhi.compute import record_coordinate_relation
    folder = tmp_path / 'calibrations'
    folder.mkdir()
    path = folder / 'camera-camchain.yaml'
    original = '# original calibration\n' + yaml.safe_dump(calibration())
    path.write_text(original, encoding='utf-8')
    original_bytes = path.read_bytes()
    record_coordinate_relation(tmp_path)
    first = path.read_bytes()
    assert first.startswith(original_bytes.rstrip(b'\r\n'))
    data = yaml.safe_load(first)
    for name, camera in calibration().items():
        assert data[name] == camera
    record_coordinate_relation(tmp_path)
    assert path.read_bytes() == first
    data['head_local_to_calibration_body']['translation'] = [1, 0, 0]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='refusing to overwrite'):
        record_coordinate_relation(tmp_path)
