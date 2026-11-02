"""Sensor diagnostics and short-window consistency checks, not inertial ground truth."""
import numpy as np


def summarize(imu, camera_times=None, offset_s=0.):
    t = np.asarray(imu['timestamp'], dtype=float)
    a = np.column_stack([imu[k] for k in ('ax', 'ay', 'az')]).astype(float)
    g = np.column_stack([imu[k] for k in ('gx', 'gy', 'gz')]).astype(float)
    if len(t) < 2 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('IMU requires at least two strictly increasing finite timestamps')
    if a.shape != (len(t), 3) or g.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(g).all():
        raise ValueError('Invalid IMU measurements')
    an, gn, dt = np.linalg.norm(a, axis=1), np.linalg.norm(g, axis=1), np.diff(t)
    warnings = []
    if (dt > .05).any():
        warnings.append('internal_gap_over_50ms')
    if not 7 < np.median(an) < 12:
        warnings.append('acceleration_norm_unusual_check_units_or_motion')
    if (an > 60).any() or (gn > 20).any():
        warnings.append('large_sensor_values')
    frozen = np.all(np.diff(a, axis=0) == 0, axis=1) & np.all(np.diff(g, axis=0) == 0, axis=1)
    longest = current = 0.
    for same, period in zip(frozen, dt):
        current = current + period if same else 0.
        longest = max(longest, current)
    if longest > .5:
        warnings.append('constant_six_axis_values_over_500ms')
    result = dict(status='warning' if warnings else 'no_obvious_sensor_anomaly', warnings=warnings,
                  sample_count=len(t), duration_s=float(t[-1]-t[0]),
                  median_rate_hz=float(1/np.median(dt)), max_gap_s=float(dt.max()),
                  gap_end_times_s=t[1:][dt > .05].tolist(), longest_identical_run_s=float(longest),
                  accel_norm_median_m_s2=float(np.median(an)), accel_norm_max_m_s2=float(an.max()),
                  gyro_norm_median_rad_s=float(np.median(gn)), gyro_norm_max_rad_s=float(gn.max()),
                  accel_axis_mean=a.mean(axis=0).tolist(), gyro_axis_mean=g.mean(axis=0).tolist(),
                  large_value_times_s=t[(an > 60) | (gn > 20)].tolist(),
                  thresholds=dict(max_gap_s=.05, large_accel_m_s2=60., large_gyro_rad_s=20., frozen_s=.5),
                  note='Raw sensor statistics, no bias/scale calibration or stationary ground truth assumed.')
    if camera_times is not None:
        c = np.asarray(camera_times, dtype=float) + offset_s
        result['coverage'] = dict(offset_s=float(offset_s), frames_before_imu=int((c < t[0]).sum()),
                                  frames_after_imu=int((c > t[-1]).sum()),
                                  note='Boundary coverage only; no extrapolation, internal gaps counted separately.')
    return result


def compare(imu, data, quality):
    t = np.asarray(imu['timestamp'], dtype=float)
    a = np.column_stack([imu[k] for k in ('ax', 'ay', 'az')]).astype(float)
    g = np.column_stack([imu[k] for k in ('gx', 'gy', 'gz')]).astype(float)
    indices = np.flatnonzero(data['status'] == 'valid')
    matrices = data['T_world_imu']
    events, compared, quiet, mismatch, skipped = [], 0, 0, 0, 0
    flagged = set(quality['jump_frames']) | set(quality['abnormal_speed_frames'])
    residuals = []
    for i, j in zip(indices[:-1], indices[1:]):
        # Compare adjacent video frames only; never bridge missing VIO states.
        lo, hi = float(data['timestamp_imu'][i]), float(data['timestamp_imu'][j])
        camera_dt = float(data['timestamp_camera'][j] - data['timestamp_camera'][i])
        if j != i+1 or not t[0] <= lo < hi <= t[-1] or hi-lo > .2 or camera_dt <= 0:
            skipped += 1
            continue
        start, end = max(0, np.searchsorted(t, lo, side='right')-1), min(len(t)-1, np.searchsorted(t, hi))
        if (np.diff(t[start:end+1]) > .05).any():
            skipped += 1
            continue
        sample_t = np.r_[lo, t[(t > lo) & (t < hi)], hi]
        gyro = np.column_stack([np.interp(sample_t, t, g[:, k]) for k in range(3)])
        accel = np.column_stack([np.interp(sample_t, t, a[:, k]) for k in range(3)])
        gyro_norm = np.linalg.norm(gyro, axis=1)
        # Integral of angular speed is an upper bound on net rotation, not a vector rotation estimate.
        gyro_angle = float(np.sum((gyro_norm[:-1]+gyro_norm[1:])*.5*np.diff(sample_t)))
        r = matrices[i, :3, :3].T @ matrices[j, :3, :3]
        angle = float(np.arccos(np.clip((np.trace(r)-1)/2, -1, 1)))
        excess = max(0., angle-gyro_angle)
        residuals.append(excess)
        rotation_mismatch = excess > .05  # ~2.9 degrees in one frame, heuristic only
        mismatch += rotation_mismatch
        # Local low-motion evidence must persist for >=0.5 s; it does not prove zero velocity.
        window = (t >= (lo+hi)/2-.25) & (t <= (lo+hi)/2+.25)
        local_t = t[window]
        low_motion = (len(local_t) >= 3 and local_t[-1]-local_t[0] >= .4
                      and np.max(np.diff(local_t)) <= .05
                      and np.max(np.linalg.norm(g[window], axis=1)) < .05
                      and np.max(np.std(a[window], axis=0)) < .15
                      and abs(np.median(np.linalg.norm(a[window], axis=1))-9.81) < .8)
        speed = float(np.linalg.norm(matrices[j, :3, 3]-matrices[i, :3, 3])/camera_dt)
        conflict = bool(low_motion and speed > 1.)
        quiet += conflict
        if j in flagged or rotation_mismatch or conflict:
            events.append(dict(frame_index=int(j), timestamp_camera_s=float(data['timestamp_camera'][j]),
                               timestamp_imu_s=hi, vio_speed_m_s=speed,
                               vio_rotation_deg=float(np.degrees(angle)), gyro_integral_deg=float(np.degrees(gyro_angle)),
                               rotation_excess_deg=float(np.degrees(excess)), rotation_mismatch=bool(rotation_mismatch),
                               low_imu_motion=bool(low_motion), low_motion_high_vio_speed=conflict,
                               nearby_imu_gap=bool((np.diff(t[start:end+1]) > .02).any()),
                               large_imu_value=bool((np.linalg.norm(accel, axis=1)>60).any() or (gyro_norm>20).any())))
        compared += 1
    return dict(status='warning' if mismatch or quiet else ('no_obvious_inconsistency' if compared else 'insufficient_overlap'),
                compared_intervals=compared, skipped_intervals=skipped,
                rotation_mismatch_count=int(mismatch), low_motion_high_speed_count=int(quiet),
                max_rotation_excess_deg=float(np.degrees(max(residuals, default=0))), events=events,
                thresholds=dict(rotation_excess_rad=.05, low_motion_gyro_rad_s=.05,
                                low_motion_accel_axis_std_m_s2=.15, low_motion_vio_speed_m_s=1.),
                note='Raw IMU is uncalibrated here. Low IMU motion cannot distinguish rest from constant velocity. '
                     'No double integration or position truth; warnings do not identify the sensor as the cause.')
