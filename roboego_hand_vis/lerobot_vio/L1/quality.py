"""Structural validation and configurable trajectory plausibility checks."""
import numpy as np

DEFAULTS = dict(min_valid_ratio=.5, max_step_m=1., max_speed_m_s=5.,
                max_angular_speed_deg_s=720., max_distance_m=50.)


def check_trajectory(data, source_times, thresholds=None, *, camera_mode='stereo'):
    limits = {**DEFAULTS, **(thresholds or {})}
    if any(not np.isfinite(v) or v < 0 for v in limits.values()) or limits['min_valid_ratio'] > 1:
        raise ValueError('Invalid quality thresholds')
    times = np.asarray(source_times, dtype=float)
    n = len(times)
    if n == 0 or not np.isfinite(times).all():
        raise ValueError('Invalid source timestamps')
    if str(data['schema_version']) != 'openvins.frame_aligned.v2':
        raise ValueError('Unexpected trajectory schema')
    for key in ('frame_index', 'right_frame_index', 'timestamp_camera', 'timestamp_imu', 'status'):
        if data[key].shape != (n,):
            raise ValueError(f'Invalid {key} shape')
    if not np.array_equal(data['frame_index'], np.arange(n)):
        raise ValueError('Trajectory frame indices do not match source video')
    known = np.isfinite(data['timestamp_camera'])
    if not np.allclose(data['timestamp_camera'][known], times[known], atol=1e-6, rtol=0):
        raise ValueError('Trajectory timestamps do not match source')
    valid = data['status'] == 'valid'
    if not valid.any():
        raise ValueError('No valid poses')
    if not np.isfinite(data['timestamp_imu'][valid]).all() or not known[valid].all():
        raise ValueError('Valid poses require finite camera/IMU timestamps')
    if camera_mode != 'stereo':
        raise ValueError('Invalid camera mode')
    if camera_mode == 'stereo' and (data['right_frame_index'][valid] < 0).any():
        raise ValueError('Valid pose has no stereo pair')
    for key in ('T_world_imu', 'T_world_camera'):
        matrices = data[key]
        if matrices.shape != (n, 4, 4):
            raise ValueError(f'Invalid {key} shape')
        t = matrices[valid]
        if not np.isfinite(t).all() or not np.allclose(t[:, 3, :], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f'Invalid homogeneous matrix: {key}')
        r = t[:, :3, :3]
        if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(r), 1, atol=1e-4):
            raise ValueError(f'Invalid rotation matrix: {key}')
        if not np.isnan(matrices[~valid]).all():
            raise ValueError(f'Invalid frames must contain NaN matrices: {key}')
    indices = np.flatnonzero(valid)
    t = data['T_world_imu'][valid]
    dt = np.diff(times[valid])
    distance = np.linalg.norm(np.diff(t[:, :3, 3], axis=0), axis=1)
    rotation = t[:-1, :3, :3].transpose(0, 2, 1) @ t[1:, :3, :3]
    angles = np.degrees(np.arccos(np.clip((np.trace(rotation, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    advancing = dt > 0
    edge_indices = indices[1:][advancing]
    distance, angles, dt = distance[advancing], angles[advancing], dt[advancing]
    speed, angular = distance / dt, angles / dt
    # Position differences across missing poses use actual elapsed time.
    excursion = np.linalg.norm(t[:, :3, 3] - t[0, :3, 3], axis=1)
    ratio = float(valid.mean())
    reasons = []
    if ratio < limits['min_valid_ratio']:
        reasons.append('low_valid_pose_ratio')
    for name, values, limit in [('translation_jump', distance, limits['max_step_m']),
                                 ('abnormal_speed', speed, limits['max_speed_m_s']),
                                 ('abnormal_angular_speed', angular, limits['max_angular_speed_deg_s']),
                                 ('large_excursion_possible_drift', excursion, limits['max_distance_m'])]:
        if (values > limit).any():
            reasons.append(name)
    return dict(status='failed' if reasons else 'passed_basic_checks', reasons=reasons,
                thresholds=limits, frame_count=n, valid_pose_count=int(valid.sum()), valid_ratio=ratio,
                first_valid_frame=int(indices[0]), last_valid_frame=int(indices[-1]),
                path_length_m=float(distance.sum()), net_displacement_m=float(excursion[-1]),
                max_distance_from_start_m=float(excursion.max()),
                max_step_m=float(distance.max(initial=0)), max_speed_m_s=float(speed.max(initial=0)),
                max_angular_speed_deg_s=float(angular.max(initial=0)),
                jump_frames=edge_indices[distance > limits['max_step_m']].tolist(),
                abnormal_speed_frames=edge_indices[speed > limits['max_speed_m_s']].tolist(),
                drift_accuracy='not_measurable_without_reference',
                note='Plausibility checks do not certify accuracy; large excursion may be real motion.')
