import cv2
import numpy as np
import pytest
from roboego_hand_vis.common import stereo_quality as quality


@pytest.mark.parametrize('dy,expected', [(0, 'passed'), (48, 'warning')])
def test_matching_measures_vertical_error(dy, expected):
    image = np.random.default_rng(7).integers(0,256,(480,640,3),dtype=np.uint8)
    right = cv2.warpAffine(image,np.float32([[1,0,-12],[0,1,dy]]),(640,480))
    report = quality.evaluate(image,right)
    assert report['status'] == expected
    assert abs(report['median_vertical_error_px'] - dy) < .5


def test_untextured_frame_cannot_pass():
    image = np.zeros((240,320,3),dtype=np.uint8)
    assert quality.evaluate(image,image)['status'] == 'inconclusive'


def test_guanglun_ignores_outlier_tail_but_keeps_median_gate():
    image = np.random.default_rng(7).integers(0,256,(480,640,3),dtype=np.uint8)
    right = image.copy()
    right[300:] = cv2.warpAffine(image, np.float32([[1,0,0],[0,1,60]]), (640,480))[300:]
    strict = quality.evaluate(image, right)
    relaxed = quality.evaluate(image, right, policy=quality.GUANGLUN_POLICY)
    assert strict['p95_vertical_error_px'] > 40
    assert strict['status'] == 'warning'
    assert relaxed['status'] == 'passed'
    shifted = cv2.warpAffine(image, np.float32([[1,0,0],[0,1,48]]), (640,480))
    assert quality.evaluate(image, shifted, policy=quality.GUANGLUN_POLICY)['status'] == 'warning'
    assert quality.evaluate(np.zeros_like(image), np.zeros_like(image), policy=quality.GUANGLUN_POLICY)['status'] == 'inconclusive'


def test_cached_gate_and_blocked_report(tmp_path, monkeypatch):
    paths = [tmp_path/'left.mp4',tmp_path/'right.mp4']
    for path in paths:
        path.touch()
    class Capture:
        def read(self): return True,np.zeros((10,10,3),dtype=np.uint8)
        def release(self): pass
    monkeypatch.setattr(cv2,'VideoCapture',lambda *args: Capture())
    calls=[]
    def evaluate(*args):
        calls.append(1)
        return dict(status='passed', reason='test')
    monkeypatch.setattr(quality,'evaluate',evaluate)
    ep=tmp_path/'data/chunk-000/episode_000000'
    quality.require_stereo_quality(tmp_path,ep,*paths)
    quality.require_stereo_quality(tmp_path,ep,*paths)
    assert len(calls)==1
    paths[0].write_bytes(b'changed')
    monkeypatch.setattr(quality,'evaluate',lambda *args: dict(status='warning',reason='bad_alignment'))
    with pytest.raises(ValueError,match='bad_alignment'):
        quality.require_stereo_quality(tmp_path,ep,*paths)
    assert (tmp_path/'annotation/stereo_quality/chunk-000_episode_000000.json').exists()
