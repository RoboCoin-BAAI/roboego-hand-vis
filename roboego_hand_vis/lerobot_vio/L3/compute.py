"""Load L3 layout, materializing calibration only inside a VIO run workspace."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from roboego_hand_vis.lerobot_vio.L1.compute import run_session
from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import geometry_from_mapping


def calibration_row(session, episode):
    index = int(episode.name.removeprefix('episode_'))
    path = session / 'meta/calibrations.jsonl'
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    matches = [r for r in rows if r.get('episode_index') == index]
    if len(matches) != 1:
        raise ValueError(f'Expected one calibration for episode {index}, found {len(matches)}')
    head = matches[0]['calibrations']['ego_headband']
    # Validate the stereo model and identify left/right via their actual topics.
    geometry_from_mapping(head['camera-camchain'])
    for key in ('camera_imu-camchain-imucam', 'camera_imu-imu'):
        if not isinstance(head.get(key), dict) or not head[key]:
            raise ValueError(f'Missing calibration: {key}')
    if 'imu0' not in head['camera_imu-imu']:
        raise ValueError('No imu0 calibration; cannot use headband_imu0')
    return head


def normalize_cameras(chain):
    selected = {}
    for key, camera in chain.items():
        parts = str(camera.get('rostopic', '')).split('/')
        if len(parts) > 2 and parts[2] in ('left', 'right'):
            side = parts[2]
            if side in selected:
                raise ValueError(f'Ambiguous {side} calibration')
            selected[side] = (key, camera)
    if set(selected) != {'left', 'right'}:
        raise ValueError('Missing left/right camera topics')
    left_key, right_key = selected['left'][0], selected['right'][0]
    if not (left_key.startswith('cam') and right_key.startswith('cam') and
            int(right_key[3:]) == int(left_key[3:]) + 1):
        raise ValueError('Right camera must immediately follow left in the Kalibr chain')
    return {f'cam{i}': deepcopy(selected[side][1]) for i, side in enumerate(('left', 'right'))}


def plan_session(session, input_mode='rectified', zupt=False):
    if input_mode not in ('raw', 'rectified'):
        raise ValueError('Invalid input mode')
    session = Path(session).resolve()
    episodes = sorted(p for p in (session / 'data').glob('chunk-*/episode_*') if p.is_dir())
    if not episodes:
        raise ValueError('No LeRobot episodes found')
    plans = []
    for episode in episodes:
        head = calibration_row(session, episode)
        normalize_cameras(head['camera_imu-camchain-imucam'])
        prefix = f'{episode.parent.name}_{episode.name}_' if len(episodes) > 1 else ''
        paths = dict(imu=episode / 'observation.headband_imu0.parquet',
                     calibration_jsonl=session / 'meta/calibrations.jsonl')
        if input_mode == 'rectified':
            paths.update(manifest=session / 'calibrations' / f'{prefix}dataset_manifest.json',
                         rectify=session / 'calibrations' / f'{prefix}rectify_params.yaml')
        for side in ('left', 'right'):
            key = f'observation.images.headband_{side}_rgb' + ('_rectified' if input_mode == 'rectified' else '')
            paths[f'{side}_video'] = session / 'videos' / episode.parent.name / key / f'{episode.name}.mp4'
            paths[f'{side}_timestamps'] = episode / f'{key}.parquet'
        for key, path in paths.items():
            if not path.is_file():
                raise ValueError(f'Missing {key}: {path}')
        folder = ('vio_raw' if input_mode == 'raw' else 'vio') + ('_zupt' if zupt else '')
        plans.append(dict(session=session, episode=episode, input_mode=input_mode, **paths,
                          source_camera='observation.images.headband_left_rgb' + ('_rectified' if input_mode == 'rectified' else ''),
                          source_imu='observation.headband_imu0',
                          output=session / folder / episode.parent.name / episode.name))
    return plans


def materialize_calibration(plan, work):
    import yaml
    head = calibration_row(plan['session'], plan['episode'])
    directory = Path(work) / 'source_calibration'
    directory.mkdir(parents=True, exist_ok=True)
    adapted = dict(plan)
    groups = {'camera_imu': ('camera_imu-camchain-imucam', True),
              'imu_calibration': ('camera_imu-imu', False),
              'stereo_calibration': ('camera-camchain', True)}
    for name, (key, cameras) in groups.items():
        value = normalize_cameras(head[key]) if cameras else head[key]
        path = directory / f'{key}.yaml'
        path.write_text(yaml.safe_dump(value), encoding='utf-8')
        adapted[name] = path
    provenance = dict(path=str(plan['calibration_jsonl']),
                      episode_index=int(plan['episode'].name.removeprefix('episode_')),
                      imu_feature='observation.headband_imu0',
                      sha256=hashlib.sha256(json.dumps(head, sort_keys=True, allow_nan=False).encode()).hexdigest())
    (directory / 'source.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return adapted
