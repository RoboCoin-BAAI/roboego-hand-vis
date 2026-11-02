from types import SimpleNamespace

import pytest

from roboego_hand_vis.lerobot_headband_integrity.L1.imu import signal_quality, check_episode
from roboego_hand_vis.lerobot_headband_integrity.L3.compute import check_feature
from roboego_hand_vis.lerobot_headband_integrity.common import IMU_COLUMNS


@pytest.mark.parametrize('level', ['L1', 'L3'])
@pytest.mark.parametrize('frozen_count', [1, 3, 6])
def test_any_axis_constant_across_all_frames_fails(level, frozen_count):
    columns = {name: [1.] * 100 if i < frozen_count else [j * .01 for j in range(100)]
               for i, name in enumerate(IMU_COLUMNS)}
    table = dict(timestamp=[i * .005 for i in range(100)], **columns)
    ctx = SimpleNamespace(table=lambda *args: table, times=lambda *args: [],)
    if level == 'L1':
        checks = check_episode(ctx, None)
    else:
        checks = check_feature(ctx, None, 'observation.headband_imu0',
                               dict(dtype='float32', shape=[6], names=list(IMU_COLUMNS)), [])
    frozen = next(c for c in checks if c['name'] == 'frozen')
    assert not frozen['passed']
    assert frozen['level'] == 'fail'
    assert len(frozen['statistics']['constant_columns']) == frozen_count


@pytest.mark.parametrize('step,spike,detected', [
    (.01, 10., False), (.01, 30., True), (.5, 30., False),
])
def test_relaxed_jump_requires_absolute_and_relative_threshold(step, spike, detected):
    values = [i * step for i in range(100)]
    values[50] += spike
    stats = signal_quality({'ax': values})
    assert bool(stats['jump_columns']) is detected


def test_near_freeze_with_actual_change_is_not_fully_constant():
    values = [1.] * 2001
    values[-1] = 1.01
    stats = signal_quality({'ax': values, 'ay': [i * .001 for i in range(2001)]})
    assert stats['frozen_columns'] == ['ax']
    assert stats['constant_columns'] == []
    assert not stats['all_columns_frozen']


@pytest.mark.parametrize('level', ['L1', 'L3'])
def test_single_nearly_frozen_axis_fails(level):
    columns = {name: [j * .001 for j in range(2001)] for name in IMU_COLUMNS}
    columns['ax'] = [1.] * 2001
    columns['ax'][-1] = 1.01
    table = dict(timestamp=[i * .005 for i in range(2001)], **columns)
    ctx = SimpleNamespace(table=lambda *args: table, times=lambda *args: [])
    checks = (check_episode(ctx, None) if level == 'L1' else
              check_feature(ctx, None, 'observation.headband_imu0',
                            dict(dtype='float32', shape=[6], names=list(IMU_COLUMNS)), []))
    frozen = next(c for c in checks if c['name'] == 'frozen')
    assert not frozen['passed']
    assert frozen['level'] == 'fail'
    assert frozen['statistics']['frozen_columns'] == ['ax']
    assert frozen['statistics']['constant_columns'] == []
