"""DS geometry tests use known 3-D points, not VIO output as a reference."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from roboego_hand_vis.stereo_rectification.double_sphere import DoubleSphereCameraModel
from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import (
    geometry_from_mapping, rectify_episode, Encoder)


def setup_geometry():
    cam = dict(camera_model='ds', distortion_model='none', distortion_coeffs=[],
               intrinsics=[-.19, .62, 32., 33., 32., 24.], resolution=[64, 48])
    left, right = copy.deepcopy(cam), copy.deepcopy(cam)
    left['rostopic'] = '/fisheye/left/image_raw'
    right['rostopic'] = '/fisheye/right/image_raw'
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([.015, -.025, .04]))[0]
    T[:3, 3] = [-.1, .003, .001]
    right['T_cn_cnm1'] = T.tolist()
    models = [DoubleSphereCameraModel(*c['intrinsics'][2:], *c['intrinsics'][:2]) for c in (left, right)]
    params = models[0].stereo_rectify(*models, (64, 48), T[:3, :3], T[:3, 3])
    return left, right, T, models, params


def test_epipolar_alignment_depth_and_remap_direction():
    left, right, T, models, p = setup_geometry()
    geometry_from_mapping(dict(cam0=left, cam1=right))
    rng = np.random.default_rng(13)
    points = rng.uniform([-.3, -.2, 1.], [.3, .2, 3.], (100, 3))
    right_points = points @ T[:3, :3].T + T[:3, 3]
    rectified = [points @ p['R1'].T, right_points @ p['R2'].T]
    uv = []
    for xyz, P in zip(rectified, (p['P1'], p['P2'])):
        xy = xyz @ P[:3, :3].T
        uv.append(xy[:, :2]/xy[:, 2:])
    np.testing.assert_allclose(uv[0][:, 1], uv[1][:, 1], atol=1e-10)
    disparity = uv[0][:, 0]-uv[1][:, 0]
    assert (disparity > 0).all()
    recovered = np.c_[uv[0], disparity, np.ones(len(points))] @ p['Q'].T
    np.testing.assert_allclose(recovered[:, :3]/recovered[:, 3:], rectified[0], atol=1e-10)
    for m, R, P in zip(models, (p['R1'], p['R2']), (p['P1'], p['P2'])):
        mx, my = m.generate_stereo_remap(R, P, 64, 48)
        pixel = np.array([25., 20., 1.])
        ray = R.T @ np.linalg.solve(P[:3, :3], pixel)
        expected = m.double_sphere_project(ray[None])[0]
        np.testing.assert_allclose([mx[20, 25], my[20, 25]], expected, atol=2e-6)


def test_ds_projection_domain_and_pinhole_limit():
    model = DoubleSphereCameraModel(30, 30, 32, 24, 0, 0)
    points = np.array([[.1, .2, 1], [0, 0, -1], [0, 0, 0]], dtype=float)
    uv = model.double_sphere_project(points)
    np.testing.assert_allclose(uv[0], [35, 30])
    assert (uv[1:] == -1).all()


def test_ds_encodes_and_reuses_rectified_episode(tmp_path):
    import imageio_ffmpeg
    from roboego_hand_vis.lerobot_headband_integrity.common import Context
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    episode = tmp_path / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    (tmp_path / 'meta').mkdir()
    features = {}
    left, right, T, _, _ = setup_geometry()
    for side in ('left', 'right'):
        feature = f'observation.images.head_{side}'
        features[feature] = dict(dtype='video', shape=[48, 64, 3], fps=30)
        pq.write_table(pa.table(dict(timestamp=[0., 1/30, 2/30])), episode / f'{feature}.parquet')
        encoder = Encoder(tmp_path/'videos/chunk-000'/feature/'episode_000000.mp4', (64, 48), 30, ffmpeg)
        for i in range(3):
            encoder.write(np.full((48, 64, 3), 60+i*20, dtype=np.uint8))
        encoder.close()
    (tmp_path/'meta/info.json').write_text(json.dumps(dict(fps=30, features=features)))
    ctx = Context(tmp_path)
    result = rectify_episode(ctx, episode, geometry=(left, right, T), ffmpeg=ffmpeg)
    assert result['frame_counts'] == {'left': 3, 'right': 3}
    manifest = json.loads((tmp_path/'calibrations/dataset_manifest.json').read_text())
    assert manifest['intrinsics']['left'] == manifest['intrinsics']['right']
    assert manifest['extrinsics']['baseline_m'] == pytest.approx(np.linalg.norm(T[:3, 3]))
    # Reload published feature metadata before verifying the reuse path.
    again = rectify_episode(Context(tmp_path), episode, geometry=(left, right, T), ffmpeg=ffmpeg)
    assert again['skipped_existing'] is True
