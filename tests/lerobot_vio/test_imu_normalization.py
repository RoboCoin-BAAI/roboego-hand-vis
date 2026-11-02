import copy
import pytest
from roboego_hand_vis.lerobot_vio.L1.compute import normalize_imu


def test_stable_sort_keeps_measurements_attached_and_first_duplicate():
    table = dict(timestamp=[2., 1., 2., 3.],
                 **{key: [20., 10., 99., 30.] for key in ('gx','gy','gz','ax','ay','az')})
    original = copy.deepcopy(table)
    cleaned, report = normalize_imu(table)
    assert cleaned['timestamp'] == [1., 2., 3.]
    for key in ('gx','gy','gz','ax','ay','az'):
        assert cleaned[key] == [10., 20., 30.]
    assert table == original
    assert report['duplicates_removed'] == 1
    assert report['backward_steps'] == 1


@pytest.mark.parametrize('damage', ['empty', 'nan_timestamp', 'bad_duplicate', 'missing', 'length'])
def test_invalid_imu_is_not_hidden_by_cleanup(damage):
    table = dict(timestamp=[1., 1.], **{k: [1., 2.] for k in ('gx','gy','gz','ax','ay','az')})
    if damage == 'empty':
        table = {k: [] for k in table}
    elif damage == 'nan_timestamp':
        table['timestamp'][0] = float('nan')
    elif damage == 'bad_duplicate':
        table['gx'][1] = float('inf')
    elif damage == 'missing':
        del table['az']
    else:
        table['az'] = []
    with pytest.raises(ValueError):
        normalize_imu(table)
