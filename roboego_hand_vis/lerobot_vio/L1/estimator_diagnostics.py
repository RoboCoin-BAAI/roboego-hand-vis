"""Summarize observational OpenVINS diagnostics without changing quality gates."""
import csv
import math
from collections import Counter
from pathlib import Path


def summarize(path):
    with Path(path).open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows if r['initialized'] == '1']
    def number(row, key):
        value = float(row.get(key, 'nan'))
        return value if math.isfinite(value) else None
    def norm(row, prefix):
        values = [number(row, prefix+'_'+axis) for axis in 'xyz']
        return math.sqrt(sum(v*v for v in values)) if all(v is not None for v in values) else None
    def compact(row):
        return dict(frame_index=int(row['left_frame_index']), timestamp_s=number(row, 'timestamp'),
                    speed_m_s=norm(row, 'v_post'), accel_bias_m_s2=[number(row, 'ba_'+axis) for axis in 'xyz'],
                    gyro_bias_rad_s=[number(row, 'bg_'+axis) for axis in 'xyz'],
                    gravity_in_imu_m_s2=[number(row, 'gravity_in_imu_'+axis) for axis in 'xyz'],
                    accel_bias_before_m_s2=[number(row, 'ba_pre_'+axis) for axis in 'xyz'],
                    gyro_bias_before_rad_s=[number(row, 'bg_pre_'+axis) for axis in 'xyz'],
                    covariance_before={group:[number(row, 'cov_pre_'+group+'_'+axis) for axis in 'xyz']
                                       for group in ('theta', 'p', 'v', 'bg', 'ba')},
                    covariance_after={group:[number(row, 'cov_post_'+group+'_'+axis) for axis in 'xyz']
                                      for group in ('theta', 'p', 'v', 'bg', 'ba')},
                    current_tracks=int(row['current_tracks']),
                    position_update_m=number(row, 'position_update_m'), velocity_update_m_s=number(row, 'velocity_update_m_s'),
                    msckf_candidates=int(row['msckf_candidates']), msckf_accepted=int(row['msckf_accepted']),
                    msckf_tri_failed=int(row['msckf_tri_failed']), msckf_chi2_rejected=int(row['msckf_chi2_rejected']))
    counts = ('msckf_candidates', 'msckf_short', 'msckf_tri_failed', 'msckf_refine_failed',
              'msckf_chi2_rejected', 'msckf_accepted', 'msckf_ekf_applied', 'slam_candidates',
              'slam_short', 'slam_chi2_rejected', 'slam_accepted', 'slam_ekf_calls',
              'slam_init_candidates', 'slam_init_short', 'slam_init_tri_failed',
              'slam_init_refine_failed', 'slam_init_rejected', 'slam_initialized')
    def updated(row):
        return int(row['msckf_ekf_applied']) > 0 or int(row['slam_ekf_calls']) > 0 or int(row['slam_initialized']) > 0
    best, current, previous = 0., 0., None
    for row in rows:
        time = float(row['timestamp'])
        if previous is not None and not updated(row):
            current += time-previous
        else:
            current = 0.
        best = max(best, current)
        previous = time
    eligible = [r for r in rows if number(r, 'position_update_m') is not None]
    zupt_available = bool(rows and 'zupt_applied' in rows[0])
    zupt_rows = [r for r in rows if r.get('zupt_applied') == '1']
    return dict(initialized_frames=len(rows), first_states=[compact(r) for r in rows[:10]],
                zupt=dict(trace_available=zupt_available,
                          considered_frames=sum(r.get('zupt_considered') == '1' for r in rows),
                          attempted_frames=sum(r.get('zupt_attempted') == '1' for r in rows),
                          applied_frames=len(zupt_rows),
                          frame_indices=[int(r['left_frame_index']) for r in zupt_rows],
                          timestamps_s=[float(r['timestamp']) for r in zupt_rows],
                          note='Applied includes a reused same-timestamp decision; attempted counts actual try_update calls. '
                               'ZUPT frames are not visual updates and may have no pre-visual snapshot.'),
                update_counts={key:sum(int(r[key]) for r in rows) for key in counts},
                frames_without_visual_update=sum(not updated(r) for r in rows),
                longest_visual_update_gap_s=best,
                largest_position_updates=[compact(r) for r in sorted(eligible, key=lambda r:float(r['position_update_m']), reverse=True)[:10]],
                note='Biases and gravity direction are estimates, not ground truth. Pre/post snapshots bracket '
                     'MSCKF, SLAM update and landmark initialization together; pre is after IMU propagation. '
                     'NaN means the visual stage was not reached. Counts are feature-update attempts, not unique features.')


def summarize_features(path):
    """Aggregate stages separately and retain the earliest update window."""
    with Path(path).open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    def statistics(values):
        values = sorted(v for v in values if math.isfinite(v))
        if not values:
            return dict(count=0, median=None, p90=None, max=None)
        def quantile(q):
            position = q*(len(values)-1)
            lower = math.floor(position)
            upper = math.ceil(position)
            fraction = position-lower
            return (1-fraction)*values[lower] + fraction*values[upper]
        return dict(count=len(values), median=quantile(.5), p90=quantile(.9), max=values[-1])
    for stage in sorted({r['stage'] for r in rows}):
        group = [r for r in rows if r['stage'] == stage]
        frames = sorted({int(r['left_frame_index']) for r in group})
        def frame_summary(index):
            selected = [r for r in group if int(r['left_frame_index']) == index]
            return dict(frame_index=index, outcomes=dict(Counter(r['outcome'] for r in selected)),
                        raw_pixel_rms=statistics([float(r['raw_pixel_rms']) for r in selected]),
                        chi2_to_threshold=statistics([float(r['chi2'])/float(r['chi2_threshold']) for r in selected
                                                      if float(r['chi2_threshold']) > 0]))
        result[stage] = dict(attempts=len(group), outcomes=dict(Counter(r['outcome'] for r in group)),
                            triangulation_failures=dict(Counter(r['tri_reason'] for r in group if r['outcome']=='triangulation_failed')),
                            refinement_failures=dict(Counter(r['refine_reason'] for r in group if r['outcome']=='refinement_failed')),
                            first_update_frames=[frame_summary(i) for i in frames[:12]],
                            raw_pixel_rms=statistics([float(r['raw_pixel_rms']) for r in group]))
    return dict(stages=result, note='Raw RMS is per pixel coordinate before projection. Chi-square ratios '
                'use the actual projected residual and configured threshold; >1 means rejected when evaluated. '
                'SLAM initialization gate values are not exposed, so its chi-square fields remain unavailable.')
