from roboego_hand_vis.lerobot_vio.L1.compute import validate_times, runtime_camera_samples


def test_runtime_camera_order_preserves_frame_identity():
    times = [0., 1., 1., .5, 2.]
    original = list(times)
    assert validate_times(times, 'camera', strictly_increasing=False) == times
    samples, report = runtime_camera_samples(times)
    assert [(s.frame_index, s.timestamp) for s in samples] == [(0,0.), (1,1.), (4,2.)]
    assert report['skipped_frame_indices'] == [2,3]
    assert times == original
