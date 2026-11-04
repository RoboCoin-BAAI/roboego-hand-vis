"""Write relative device poses using the multi-frequency LeRobot layout."""
import json
from pathlib import Path

import numpy as np

FEATURE = 'observation.relative_head_pose'
STATUS_FEATURE = 'observation.relative_head_pose_status'
NAMES = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']


def quaternion(rotation):
    # Symmetric eigen formulation, Hamilton xyzw, stable at 180 degrees.
    r = rotation
    k = np.array([[r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
                  [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
                  [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def stage_features(plan, data, staging, quality):
    import pyarrow as pa
    import pyarrow.parquet as pq

    camera_pose = plan.get('publish_left_camera_pose', False)
    feature_key = 'observation.left_camera_pose' if camera_pose else FEATURE
    status_key = 'observation.left_camera_pose_status' if camera_pose else STATUS_FEATURE
    session = plan['session']
    source = pq.read_table(plan['left_timestamps'])
    times = source['timestamp'].to_numpy()
    valid = data['status'] == 'valid'
    matrices = data['T_world_camera' if camera_pose else 'T_world_imu']
    origin = matrices[np.flatnonzero(valid)[0]]
    values = np.full((len(times), 7), np.nan)
    previous = None
    for index in np.flatnonzero(valid):
        relative = np.linalg.inv(origin) @ matrices[index]
        q = quaternion(relative[:3, :3])
        if previous is not None and np.dot(previous, q) < 0:
            q = -q
        previous = q
        values[index] = [*relative[:3, 3], *q]
    episode_relative = plan['episode'].relative_to(session)
    folder = staging / episode_relative
    folder.mkdir(parents=True)
    base = dict(timestamp=pa.array(times, type=pa.float64()), episode_index=source['episode_index'])
    pq.write_table(pa.table({**base, **{name: pa.array(values[:, i], type=pa.float64()) for i, name in enumerate(NAMES)}}),
                   folder / f'{feature_key}.parquet')
    statuses = ['valid', 'initializing', 'imu_unavailable', 'state_not_advanced',
                'invalid_state', 'not_processed', 'not_requested', 'no_stereo_pair', 'not_estimated']
    codes = {name: i for i, name in enumerate(statuses)}
    if not set(data['status'].tolist()).issubset(codes):
        raise ValueError('Unknown trajectory status')
    pq.write_table(pa.table({**base, 'status': pa.array([codes[s] for s in data['status']], type=pa.int64())}),
                   folder / f'{status_key}.parquet')
    info = json.loads((session / 'meta/info.json').read_text(encoding='utf-8'))
    feature = dict(dtype='float64', shape=[7], names=NAMES, fps=info['fps'],
                   frame_convention='T_first_valid_left_camera_current_left_camera' if camera_pose else 'T_first_valid_imu_current_imu', translation_unit='m',
                   quaternion_convention='Hamilton xyzw', source='OpenVINS',
                   invalid_value='NaN', reference='per-episode first valid left camera pose' if camera_pose else 'per-episode first valid IMU pose')
    for key, spec in [(feature_key, feature), (status_key, dict(dtype='int64', shape=[1], names=['status'], fps=info['fps'], status_codes=codes))]:
        existing = info['features'].get(key)
        if existing is not None and existing != spec:
            raise ValueError(f'Existing feature has a different definition: {key}')
        info['features'][key] = spec
    (staging / 'meta').mkdir(exist_ok=True)
    (staging / 'meta/info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    # Preserve unrelated statistics and exclude missing poses from pose statistics.
    stats_path = session / 'meta/episodes_stats.jsonl'
    records = [json.loads(line) for line in stats_path.read_text(encoding='utf-8').splitlines() if line.strip()] if stats_path.exists() else []
    episode_index = int(source['episode_index'][0].as_py())
    record = next((row for row in records if row['episode_index'] == episode_index), None)
    if record is None:
        record = dict(episode_index=episode_index, stats={})
        records.append(record)
    def statistics(v):
        return dict(min=v.min(axis=0).tolist(), max=v.max(axis=0).tolist(),
                    mean=v.mean(axis=0).tolist(), std=v.std(axis=0).tolist(), count=[len(v)])
    record['stats'][feature_key] = statistics(values[valid])
    record['stats'][status_key] = statistics(np.array([codes[s] for s in data['status']])[:, None])
    (staging / 'meta/episodes_stats.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in records), encoding='utf-8')
    metadata = dict(feature=feature_key, source_camera=plan.get('source_camera', 'observation.images.head_left_rectified'),
                    device_frame='rectified left camera' if camera_pose else 'IMU (head-mounted device); no anatomical head transform assumed',
                    reference_frame_index=int(np.flatnonzero(valid)[0]), reference_T_world_imu=origin.tolist(),
                    status_codes=codes, quality=quality,
                    timestamp='left camera acquisition time in dataset seconds; pose is camera-aligned IMU state')
    if camera_pose:
        metadata['reference_T_world_camera'] = metadata.pop('reference_T_world_imu')
        metadata['timestamp'] = 'left camera acquisition time; left camera pose from T_world_camera'
    return metadata
