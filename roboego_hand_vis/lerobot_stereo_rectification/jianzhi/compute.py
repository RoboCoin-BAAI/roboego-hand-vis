"""Adapt Jianzhi DS parameters and body-frame poses to the shared engine."""
from copy import deepcopy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from roboego_hand_vis.lerobot_headband_integrity.common import Context
from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import (
    geometry_from_mapping, rectify_episode,
)

CAMERAS = {'left': 'head_left', 'right': 'head_right'}

HEAD_LOCAL_TO_BODY = {
    'source_frame': 'head_vio_local',
    'target_frame': 'calibration_body',
    'convention': 'column vectors: p_body = R @ p_head_local + t',
    'rotation': [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
    'translation': [0, 0, 0],
    'translation_units': 'm',
    'status': 'inferred',
    'origin_assumption': 'head/VIO local origin coincides with calibration body origin',
    'projection_chain': [
        'p_head_local = R_world_head.T @ (p_hand_world - t_world_head)',
        'p_body = rotation @ p_head_local + translation',
        'p_camera = R_body_camera.T @ (p_body - t_body_camera)',
        'pixel = DS_project(p_camera)',
    ],
    'evidence': {
        'session': '001301e26b8c4ace844c7b2d12eacf9d',
        'validated_interval_s': [0, 5],
        'manual_wrist_frames_zero_based': [15, 45, 75],
        'mean_wrist_error_px': 23.44686260940558,
        'method': 'signed-axis candidate selected using six manual wrist clicks; visual finger overlay review',
        'limitations': 'Not independently verified for substantial head motion or other sessions; not an official calibration.',
    },
}


def record_coordinate_relation(session):
    """Append inferred frame metadata without rewriting original camera entries."""
    import os
    import tempfile
    import yaml

    path = Path(session) / 'calibrations/camera-camchain.yaml'
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode('utf-8-sig'))
    geometry_from_jianzhi(data)  # Never annotate an invalid calibration.
    key = 'head_local_to_calibration_body'
    if key in data:
        if data[key] != HEAD_LOCAL_TO_BODY:
            raise ValueError('Existing head/body relation differs; refusing to overwrite')
        return
    addition = yaml.safe_dump({key: HEAD_LOCAL_TO_BODY}, sort_keys=False).encode('utf-8')
    payload = raw.rstrip(b'\r\n') + b'\n\n' + addition
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def body_transform(camera):
    transform = np.asarray(camera['T_b_c_matrix'], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-4)):
        raise ValueError('Invalid Jianzhi T_b_c_matrix rigid transform')
    return transform


def geometry_from_jianzhi(data):
    # T_b_c maps camera coordinates into the common body frame.
    # x_right = inv(T_body_right) @ T_body_left @ x_left.
    pair = {name: deepcopy(data[name]) for name in CAMERAS.values()}
    transform = np.linalg.inv(body_transform(pair['head_right'])) @ body_transform(pair['head_left'])
    for camera in pair.values():
        if (camera.get('camera_model'), camera.get('distortion_model')) != ('pinhole', 'ds'):
            raise ValueError('Expected Jianzhi pinhole/ds encoding')
        coeffs = np.asarray(camera['distortion_coeffs'], dtype=float)
        intrinsics = np.asarray(camera['intrinsics'], dtype=float)
        if (coeffs.shape != (6,) or intrinsics.shape != (4,)
                or not np.isfinite(coeffs).all()
                or not np.allclose(coeffs[:4], intrinsics, rtol=1e-8, atol=1e-8)):
            raise ValueError('Expected DS coefficients [fx, fy, cx, cy, xi, alpha] matching intrinsics')
        camera.update(camera_model='ds', distortion_model='none', distortion_coeffs=[],
                      intrinsics=[*coeffs[4:].tolist(), *coeffs[:4].tolist()])
    pair['head_right'].update(T_cn_cnm1=transform.tolist(), extrinsics_reference_camera='head_left')
    return geometry_from_mapping(pair)


def load_geometry(session):
    import yaml

    root = Path(session)
    path = root / 'calibrations/camera-camchain.yaml'
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode('utf-8-sig'))
    return geometry_from_jianzhi(data), dict(
        path=path.relative_to(root).as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
        transform_convention='T_right_left = inverse(T_b_right) @ T_b_left',
        source_ds_order=['fx', 'fy', 'cx', 'cy', 'xi', 'alpha'])


def process_session(session, options):
    ctx = Context(Path(session))
    if not options.get('dry_run', False):
        record_coordinate_relation(ctx.root)
        (ctx.root / 'calibrations/jianzhi-camera-calibration.json').unlink(missing_ok=True)
        (ctx.root / 'meta/jianzhi_quality_report.json').unlink(missing_ok=True)
    options = dict(options)
    identity = hashlib.sha256(str(ctx.root).encode()).hexdigest()
    options['work_root'] = Path(tempfile.gettempdir()) / 'jianzhi-rectification' / identity
    options['write_frame_mapping'] = False
    options['full_decode_validation'] = False
    if not options.get('dry_run', False):
        for path in (ctx.root / 'calibrations').glob('*frame_mapping.parquet'):
            if path.name == 'frame_mapping.parquet' or path.name.startswith('chunk-'):
                path.unlink()
    geometry, provenance = load_geometry(ctx.root)
    for camera in CAMERAS.values():
        feature = f'observation.images.{camera}'
        if ctx.info.get('features', {}).get(feature, {}).get('dtype') != 'video':
            raise ValueError(f'Missing video feature: {feature}')
        for episode in ctx.episodes:
            if not (episode / f'{feature}.parquet').is_file():
                raise ValueError(f'Missing timestamp feature: {episode / feature}')
    return [rectify_episode(ctx, episode, camera_names=CAMERAS, geometry=geometry,
                           calibration_source=provenance, **options) for episode in ctx.episodes]
