import numpy as np
import pytest
from roboego_hand_vis.lerobot_vio.L1.imu_diagnostics import summarize, compare
from test_quality import trajectory


def sensor():
    t = np.arange(0, 2.001, .005)
    return dict(timestamp=t, ax=t*0, ay=t*0, az=t*0+9.81, gx=t*0, gy=t*0, gz=t*0)


def test_coverage_short_boundary_missing():
    imu = sensor()
    result = summarize(imu, [-.1, 0., 1., 2., 2.1])
    assert result['coverage']['frames_before_imu'] == 1
    assert result['coverage']['frames_after_imu'] == 1
    assert result['median_rate_hz'] == pytest.approx(200)


def test_gap_and_invalid_sensor():
    imu = sensor()
    imu = {k: np.delete(v, np.arange(50, 80)) for k, v in imu.items()}
    assert 'internal_gap_over_50ms' in summarize(imu)['warnings']
    imu['ax'][0] = np.nan
    with pytest.raises(ValueError):
        summarize(imu)


def test_stationary_vio_and_motion_conflict():
    data = trajectory(3)
    data['timestamp_camera'] = np.array([.8, .8333333, .8666667])
    data['timestamp_imu'] = data['timestamp_camera'].copy()
    quality = dict(jump_frames=[], abnormal_speed_frames=[2])
    assert compare(sensor(), data, quality)['low_motion_high_speed_count'] == 0
    data['T_world_imu'][2, 0, 3] = 1.
    result = compare(sensor(), data, quality)
    assert result['low_motion_high_speed_count'] == 1
    assert result['events'][0]['frame_index'] == 2


def test_rotation_and_no_extrapolation():
    data = trajectory(3)
    data['timestamp_camera'] = np.array([.8, .8333333, .8666667])
    data['timestamp_imu'] = data['timestamp_camera'].copy()
    data['T_world_imu'][1, :3, :3] = np.diag([-1., -1., 1.])
    quality = dict(jump_frames=[], abnormal_speed_frames=[])
    assert compare(sensor(), data, quality)['rotation_mismatch_count'] == 2
    data['timestamp_imu'] += 10
    result = compare(sensor(), data, quality)
    assert result['compared_intervals'] == 0
    assert result['status'] == 'insufficient_overlap'
