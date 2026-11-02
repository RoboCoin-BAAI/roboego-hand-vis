from types import SimpleNamespace
from unittest.mock import patch
import json
import pytest
from roboego_hand_vis.lerobot_headband_integrity.L1.timestamps import check_camera
from roboego_hand_vis.lerobot_headband_integrity.L3.compute import check_feature


@pytest.mark.parametrize('times,level', [([1., 1.], 'warning'), ([2., 1.], 'warning'),
                                       ([float('nan'), 1.], 'fail'), ([], 'fail')])
def test_l1_timestamp_policy(times, level):
    ctx = SimpleNamespace(times=lambda *args: times, fps=lambda *args: 30.,
                          table=lambda *args: {'timestamp': times})
    checks = check_camera(ctx, None, 'head_left')
    item = next(c for c in checks if c['name'] == 'monotonic')
    assert not item['passed'] and item['level'] == level
    if level == 'warning':
        assert not any(not c['passed'] and c['level'] == 'fail' for c in checks)


@pytest.mark.parametrize('times,level', [([1.,1.], 'warning'), ([2.,1.], 'warning'),
                                       ([float('nan'),1.], 'fail'), ([], 'fail')])
def test_l3_video_timestamp_policy(times, level):
    ctx = SimpleNamespace(table=lambda *args: {'timestamp': times}, ffprobe='ffprobe',
                          video=lambda *args: 'video.mp4', fps=lambda *args: 30.)
    probe = SimpleNamespace(stderr='', stdout=json.dumps({'streams': [dict(
        width=64, height=48, nb_read_frames=max(1,len(times)), avg_frame_rate='30/1')]}))
    with patch('subprocess.run', return_value=probe):
        checks = check_feature(ctx, None, 'observation.images.headband_left_rgb',
                               dict(dtype='video',shape=[48,64,3]), [])
    item = next(c for c in checks if c['name'] == 'timestamps')
    assert not item['passed'] and item['level'] == level
