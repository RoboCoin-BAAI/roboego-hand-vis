"""Adapt L3 JSONL calibration and feature names to the shared stereo engine."""
import hashlib
import json
from pathlib import Path

from roboego_hand_vis.lerobot_headband_integrity.common import Context
from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import (
    geometry_from_mapping, rectify_episode,
)

CAMERAS = {'left': 'headband_left_rgb', 'right': 'headband_right_rgb'}


def load_geometry(session, episode):
    """Select exactly one episode and derive the pair from Kalibr rostopics."""
    path = Path(session) / 'meta/calibrations.jsonl'
    index = int(Path(episode).name.removeprefix('episode_'))
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'Calibration row {number} must be an object')
        if row.get('episode_index') == index:
            rows.append(row)
    if len(rows) != 1:
        raise ValueError(f'Expected one calibration for episode {index}, found {len(rows)}')
    chain = rows[0]['calibrations']['ego_headband']['camera-camchain']
    if not isinstance(chain, dict):
        raise ValueError('camera-camchain must be a mapping')
    # Geometry validation checks unique left/right topics, consecutive cam keys,
    # model, resolution, rotation and nonzero baseline. Never use the outer pair.
    geometry = geometry_from_mapping(chain)
    provenance = dict(path='meta/calibrations.jsonl', episode_index=index,
                      key='calibrations.ego_headband.camera-camchain',
                      sha256=hashlib.sha256(json.dumps(chain, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest())
    return geometry, provenance


def process_session(session, options):
    ctx = Context(Path(session))
    for camera in CAMERAS.values():
        feature = f'observation.images.{camera}'
        if ctx.info.get('features', {}).get(feature, {}).get('dtype') != 'video':
            raise ValueError(f'Missing video feature: {feature}')
    # Validate every episode calibration before publishing any episode output.
    geometries = [load_geometry(ctx.root, episode) for episode in ctx.episodes]
    results = []
    for episode, (geometry, provenance) in zip(ctx.episodes, geometries):
        for camera in CAMERAS.values():
            timestamps = episode / f'observation.images.{camera}.parquet'
            if not timestamps.is_file():
                raise ValueError(f'Missing timestamp feature: {timestamps}')
        results.append(rectify_episode(ctx, episode, camera_names=CAMERAS,
                       geometry=geometry, calibration_source=provenance, **options))
    return results
