from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roboego_hand_vis.lerobot_headband_integrity.L3.compute import check_stereo_alignment


@pytest.mark.parametrize('dy,status', [(0, 'passed'), (48, 'warning')])
@pytest.mark.parametrize('dataset', ['L1', 'L3'])
def test_rectified_first_frames_measure_vertical_alignment(monkeypatch, tmp_path, dy, status, dataset):
    from roboego_hand_vis.lerobot_stereo_rectification.L3 import compute as adapter
    from roboego_hand_vis.stereo_rectification.equidistant import EquidistantCameraModel

    camera = dict(camera_model='pinhole', intrinsics=[300., 300., 320., 240.],
                  distortion_coeffs=[0.] * 4, resolution=[640, 480])
    transform = np.eye(4)
    transform[0, 3] = -.1
    monkeypatch.setattr(adapter, 'load_geometry', lambda *args: (
        (camera, camera, transform), {'episode_index': 0}))
    # Deterministic maps exercise frame selection and cropping without lens effects.
    params = dict(R1=np.eye(3), R2=np.eye(3),
                  P1=np.array([[300., 0, 320, 0], [0, 300, 240, 0], [0, 0, 1, 0]]),
                  P2=np.array([[300., 0, 320, -30], [0, 300, 240, 0], [0, 0, 1, 0]]),
                  Q=np.eye(4))
    monkeypatch.setattr(EquidistantCameraModel, 'stereo_rectify', lambda *a, **k: params)
    x, y = np.meshgrid(np.arange(640, dtype=np.float32), np.arange(480, dtype=np.float32))
    monkeypatch.setattr(EquidistantCameraModel, 'generate_stereo_remap', lambda *a: (x, y))
    image = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    right = cv2.warpAffine(image, np.float32([[1, 0, -12], [0, 1, dy]]), (640, 480))
    captures = []

    class Capture:
        def __init__(self, path):
            self.frame = image if 'left' in path else right
            self.reads = 0
            self.released = False
            captures.append(self)

        def read(self):
            self.reads += 1
            return True, self.frame

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, 'VideoCapture', Capture)
    if dataset == 'L3' and dy == 0:
        # A large tail error alone must no longer block L3 QC.
        monkeypatch.setattr(np, 'percentile', lambda values, q: 121.38)
    episode = Path('data/chunk-000/episode_000000')
    ctx = SimpleNamespace(root=tmp_path, episodes=[episode],
                          video=lambda ep, cam: Path(cam + '.mp4'))
    if dataset == 'L1':
        from roboego_hand_vis.lerobot_headband_integrity.L1 import stereo
        from roboego_hand_vis.lerobot_stereo_rectification.L1 import compute as rectifier
        (tmp_path / 'calibrations').mkdir()
        (tmp_path / 'calibrations/camera-camchain.yaml').write_text('test')
        monkeypatch.setattr(rectifier, 'load_geometry', lambda path: (camera, camera, transform))
        check, = stereo.check(ctx)
    else:
        check, = check_stereo_alignment(ctx, episode)
    assert check['passed'] == (status == 'passed')
    assert check['statistics']['status'] == status
    assert check['statistics']['frame_index'] == (9 if dataset == 'L3' else 0)
    assert check['statistics']['policy']['max_p95_px'] == (None if dataset == 'L3' else 40.)
    assert abs(check['statistics']['median_vertical_error_px'] - dy) < .5
    if dataset == 'L3' and dy == 0:
        assert check['statistics']['p95_vertical_error_px'] == 121.38
    assert len(captures) == 2
    assert all(c.reads == (10 if dataset == 'L3' else 1) and c.released for c in captures)


def test_l1_missing_calibration_fails_each_episode(tmp_path):
    from roboego_hand_vis.lerobot_headband_integrity.L1 import stereo
    ctx = SimpleNamespace(root=tmp_path, episodes=[
        Path('data/chunk-000/episode_000000'), Path('data/chunk-000/episode_000001')])
    checks = stereo.check(ctx)
    assert len(checks) == 2
    assert all(not c['passed'] and c['level'] == 'fail' for c in checks)
    assert 'episode_000001/readable' in checks[1]['name']
