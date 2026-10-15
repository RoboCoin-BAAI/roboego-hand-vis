"""Validation for the public VIO delivery report."""

from pathlib import Path


def validate_report(report: dict, *, session: Path, input_mode: str) -> int:
    """Validate execution and input identity without asserting trajectory accuracy."""
    if report.get('schema_version') != 'roboego.vio.delivery.v2':
        raise ValueError('Unsupported VIO delivery report schema')
    if (report.get('status') != 'complete' or report.get('execution_status') != 'complete'
            or type(report.get('returncode')) is not int or report['returncode'] != 0):
        raise ValueError('VIO delivery did not report successful execution')
    count = report.get('pose_count')
    if type(count) is not int or count <= 0:
        raise ValueError('VIO delivery contains no valid poses')
    inputs = report.get('input', {})
    if not isinstance(inputs, dict) or inputs.get('session') != str(session):
        raise ValueError('VIO delivery session does not match the current task')
    arguments = inputs.get('arguments', [])
    if not isinstance(arguments, list) or not all(isinstance(arg, str) for arg in arguments):
        raise ValueError('VIO delivery arguments must be a string list')
    modes = []
    for index, arg in enumerate(arguments):
        if arg == '--input_mode':
            modes.append(arguments[index + 1] if index + 1 < len(arguments) else None)
        elif arg.startswith('--input_mode='):
            modes.append(arg.split('=', 1)[1])
    if modes != [input_mode]:
        raise ValueError('VIO delivery input mode does not match the current task')
    artifacts = report.get('artifacts', {})
    if not isinstance(artifacts, dict) or artifacts.get('trajectory') != 'trajectory.npz':
        raise ValueError('VIO delivery must reference trajectory.npz in its output directory')
    offline = report.get('offline', {})
    if (not isinstance(offline, dict) or offline.get('status') != 'completed'
            or type(offline.get('pose_count')) is not int or offline['pose_count'] != count):
        raise ValueError('VIO offline execution or pose count does not match the delivery')
    alignment = report.get('frame_alignment')
    if alignment is not None and (not isinstance(alignment, dict)
                                 or type(alignment.get('valid_pose_count')) is not int
                                 or alignment['valid_pose_count'] != count):
        raise ValueError('VIO valid frame count does not match the delivery')
    return count


def validate_delivery(output: Path, *, session: Path, input_mode: str) -> int:
    import json
    import numpy as np

    output = Path(output)
    for name in ('report.json', 'trajectory.npz'):
        if (output/name).is_symlink() or not (output/name).is_file():
            raise ValueError(f'Missing or symlinked VIO delivery file: {name}')
    report = json.loads((output/'report.json').read_text())
    count = validate_report(report, session=session, input_mode=input_mode)
    with np.load(output/'trajectory.npz', allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}
    for key, expected in [('schema_version', 'openvins.frame_aligned.v2'),
                          ('quality_status', report.get('quality_status'))]:
        if key not in a or a[key].shape != () or a[key].dtype.kind != 'U' or a[key].item() != expected:
            raise ValueError(f'Invalid VIO NPZ {key}')
    if 'frame_index' not in a or a['frame_index'].ndim != 1:
        raise ValueError('Invalid VIO frame index array')
    n = len(a['frame_index'])
    for key in ('frame_index', 'right_frame_index', 'timestamp_camera', 'timestamp_imu', 'status'):
        if key not in a or a[key].shape != (n,):
            raise ValueError(f'Invalid VIO array shape: {key}')
    if (n == 0 or a['frame_index'].dtype.kind not in 'iu'
            or not np.array_equal(a['frame_index'], np.arange(n))
            or a['right_frame_index'].dtype.kind not in 'iu'
            or np.any(a['right_frame_index'] < -1)):
        raise ValueError('Invalid VIO frame indices')
    if a['status'].dtype.kind != 'U' or np.any(a['status'] == ''):
        raise ValueError('Invalid VIO frame status')
    valid = a['status'] == 'valid'
    if int(valid.sum()) != count or np.any(a['right_frame_index'][valid] < 0):
        raise ValueError('VIO valid pose count or stereo indices do not match report')
    for key in ('timestamp_camera', 'timestamp_imu'):
        if a[key].dtype.kind != 'f' or not np.isfinite(a[key][valid]).all():
            raise ValueError(f'Invalid VIO timestamps: {key}')
        if np.any(np.diff(a[key][valid]) <= 0):
            raise ValueError(f'Unordered VIO timestamps: {key}')
    for key in ('T_world_imu', 'T_world_camera'):
        if key not in a or a[key].shape != (n, 4, 4) or a[key].dtype.kind != 'f':
            raise ValueError(f'Invalid VIO transform shape: {key}')
        t = a[key][valid]
        r = t[:, :3, :3]
        if (not np.isfinite(t).all() or not np.allclose(t[:, 3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-3)
                or not np.allclose(np.linalg.det(r), 1, atol=1e-3)):
            raise ValueError(f'Invalid VIO rigid transforms: {key}')
    alignment = report.get('frame_alignment', {})
    for key, expected in [('frame_count', n), ('invalid_pose_count', n-count)]:
        if key in alignment and (type(alignment[key]) is not int or alignment[key] != expected):
            raise ValueError(f'VIO {key} does not match NPZ')
    if 'status_counts' in alignment:
        names, counts = np.unique(a['status'], return_counts=True)
        if alignment['status_counts'] != dict(zip(names.tolist(), counts.tolist())):
            raise ValueError('VIO status counts do not match NPZ')
    return count


def export_native(work: Path, output: Path) -> None:
    """Convert native intermediate JSONL into a frame-aligned public delivery."""
    import csv
    import json
    import os
    import tempfile
    import numpy as np
    from .session import load_camera_timestamps
    from .trajectory import quaternion_xyzw_to_rotation

    work, output = Path(work), Path(output)
    manifest = json.loads((work/'run_manifest.json').read_text())
    offline = json.loads((work/'trajectory/run_report.json').read_text())
    if offline.get('initialized') is not True or type(offline.get('pose_count')) is not int or offline['pose_count'] <= 0:
        raise ValueError('OpenVINS was not initialized or produced no poses')
    session = Path(manifest['session_dir'])
    frames = load_camera_timestamps(session/'timestamps/left_timestamps.txt')
    n = manifest['videos']['left']['frame_count']
    if n != len(frames) or [f.frame_index for f in frames] != list(range(n)):
        raise ValueError('Video frames and source timestamps do not align')
    camera_times = np.array([f.timestamp for f in frames], dtype=np.float64)
    a = dict(schema_version=np.array('openvins.frame_aligned.v2'), quality_status=np.array('not_validated'),
             frame_index=np.arange(n, dtype=np.int64), right_frame_index=np.full(n, -1, dtype=np.int64),
             timestamp_camera=camera_times, timestamp_imu=np.full(n, np.nan),
             status=np.full(n, 'no_stereo_pair', dtype='U32'),
             T_world_imu=np.full((n,4,4), np.nan), T_world_camera=np.full((n,4,4), np.nan))
    with Path(manifest['stereo_pairs_path']).open() as stream:
        for pair in csv.DictReader(stream):
            i, right = int(pair['left_frame_index']), int(pair['right_frame_index'])
            if not 0 <= i < n or right < 0 or a['right_frame_index'][i] != -1:
                raise ValueError('Invalid or duplicate stereo pair index')
            a['right_frame_index'][i] = right
            # Missing poses do not prove whether initialization or IMU coverage caused the omission.
            a['status'][i] = 'not_estimated'
    count = 0
    previous_index = -1
    extrinsic = np.asarray(manifest['T_imu_camera0'], dtype=np.float64)
    if extrinsic.shape != (4,4):
        raise ValueError('Invalid IMU-camera extrinsic shape')
    with (work/'trajectory/trajectory.jsonl').open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            i = row['frame_index']
            if (type(i) is not int or not previous_index < i < n
                    or row['right_frame_index'] != a['right_frame_index'][i]
                    or a['right_frame_index'][i] < 0):
                raise ValueError('Invalid or unordered native trajectory frame indices')
            if abs(float(row['timestamp_camera']) - camera_times[i]) > 1e-5:
                raise ValueError('Native trajectory timestamp does not match its source frame')
            t = np.eye(4)
            t[:3,:3] = quaternion_xyzw_to_rotation(row['orientation_world_imu_xyzw'])
            position = np.asarray(row['position_world_imu'], dtype=np.float64)
            if position.shape != (3,):
                raise ValueError('Invalid native IMU position')
            t[:3,3] = position
            a['T_world_imu'][i] = t
            a['T_world_camera'][i] = t @ extrinsic
            a['timestamp_imu'][i] = float(row['timestamp_imu'])
            a['status'][i] = 'valid'
            count += 1
            previous_index = i
    if count != offline['pose_count']:
        raise ValueError('Native pose count does not match trajectory')
    names, counts = np.unique(a['status'], return_counts=True)
    report = dict(schema_version='roboego.vio.delivery.v2', status='complete', execution_status='complete',
                  quality_status='not_validated', returncode=0, pose_count=count,
                  input={'session': str(session), 'arguments': ['--input_mode', manifest['input_mode']]},
                  artifacts={'trajectory': 'trajectory.npz'},
                  offline={**{key: value for key, value in offline.items() if key != 'artifacts'},
                           'status': 'completed'},
                  frame_alignment={'schema_version': 'openvins.frame_aligned.v2', 'frame_count': n,
                                   'valid_pose_count': count, 'invalid_pose_count': n-count,
                                   'status_counts': dict(zip(names.tolist(), counts.tolist()))})
    # Do not publish partial/corrupt output as a successful delivery.
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.delivery-', dir=output) as staging:
        staging = Path(staging)
        np.savez_compressed(staging/'trajectory.npz', **a)
        (staging/'report.json').write_text(json.dumps(report, indent=2) + '\n')
        validate_delivery(staging, session=session, input_mode=manifest['input_mode'])
        for name in ('trajectory.npz', 'report.json'):
            os.replace(staging/name, output/name)
