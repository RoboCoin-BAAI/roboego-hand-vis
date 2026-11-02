from types import SimpleNamespace
import pytest
from roboego_hand_vis.lerobot_headband_integrity.common import stereo_boundary_checks


@pytest.mark.parametrize('fps', [30., 60.])
@pytest.mark.parametrize('start,end,passed', [(0.,0.,True),(2.,2.,True),(2.01,0.,False),
                                            (0.,2.01,False),(-2.01,0.,False)])
def test_stereo_boundaries(fps, start, end, passed):
    times = {'left': [100.,110.], 'right': [100.+start/fps,110.+end/fps]}
    ctx = SimpleNamespace(times=lambda episode,camera: times[camera], fps=lambda camera: fps)
    checks = stereo_boundary_checks(ctx, None, 'left', 'right')
    assert all(c['passed'] for c in checks) is passed
    assert all(c['level'] == 'fail' for c in checks)


@pytest.mark.parametrize('right', [[], [float('nan')], [float('inf')]])
def test_invalid_boundaries_fail(right):
    ctx = SimpleNamespace(times=lambda episode,camera: [0.,1.] if camera == 'left' else right)
    assert not stereo_boundary_checks(ctx, None, 'left', 'right')[0]['passed']


def test_boundaries_use_frame_order_not_sorted_extrema():
    times = {'left': [0., 10., 1.], 'right': [0., 10., 2.]}
    ctx = SimpleNamespace(times=lambda episode,camera: times[camera], fps=lambda camera: 30.)
    checks = stereo_boundary_checks(ctx, None, 'left', 'right')
    assert checks[0]['passed']
    assert not checks[1]['passed']
