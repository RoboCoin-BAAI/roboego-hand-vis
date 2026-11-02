import csv
import json
from roboego_hand_vis.lerobot_vio.L1.estimator_diagnostics import summarize


def test_update_gap_and_initial_state(tmp_path):
    rows = []
    for i in range(4):
        row = dict(initialized='1', timestamp=str(i/30), left_frame_index=str(i),
                   current_tracks='100', position_update_m='nan' if i == 0 else str(i*.1),
                   velocity_update_m_s='nan' if i == 0 else '.01')
        for prefix in ('v_post', 'bg', 'ba', 'gravity_in_imu'):
            for axis in 'xyz':
                row[prefix+'_'+axis] = '0'
        row['gravity_in_imu_z'] = '9.81'
        row.update(zupt_considered='1' if i == 1 else '0',
                   zupt_attempted='1' if i == 1 else '0', zupt_applied='1' if i == 1 else '0')
        for key in ('msckf_candidates', 'msckf_short', 'msckf_tri_failed', 'msckf_refine_failed',
                    'msckf_chi2_rejected', 'msckf_accepted', 'msckf_ekf_applied', 'slam_candidates',
                    'slam_short', 'slam_chi2_rejected', 'slam_accepted', 'slam_ekf_calls',
                    'slam_init_candidates', 'slam_init_short', 'slam_init_tri_failed',
                    'slam_init_refine_failed', 'slam_init_rejected', 'slam_initialized'):
            row[key] = '0'
        if i == 3:
            row.update(msckf_candidates='10', msckf_tri_failed='7', msckf_chi2_rejected='2',
                       msckf_accepted='1', msckf_ekf_applied='1')
        rows.append(row)
    path = tmp_path / 'diagnostics.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(path)
    assert report['frames_without_visual_update'] == 3
    assert report['longest_visual_update_gap_s'] == 2/30
    assert report['first_states'][0]['position_update_m'] is None
    assert report['largest_position_updates'][0]['frame_index'] == 3
    assert report['update_counts']['msckf_tri_failed'] == 7
    assert report['zupt']['trace_available'] is True
    assert report['zupt']['applied_frames'] == 1
    assert report['zupt']['attempted_frames'] == 1
    assert report['zupt']['frame_indices'] == [1]
    json.dumps(report, allow_nan=False)


def test_feature_reasons_and_gate_ratios(tmp_path):
    from roboego_hand_vis.lerobot_vio.L1.estimator_diagnostics import summarize_features
    rows = []
    for outcome, reason, rms, chi2, threshold in [
        ('triangulation_failed', 'depth_below_min', 'nan', 'nan', 'nan'),
        ('chi2_rejected', 'accepted', '3', '20', '10'),
        ('accepted_for_system', 'accepted', '1', '2', '10')]:
        rows.append(dict(stage='msckf', left_frame_index='227', outcome=outcome, tri_reason=reason,
                         refine_reason='not_evaluated', raw_pixel_rms=rms, chi2=chi2, chi2_threshold=threshold))
    path = tmp_path / 'features.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = summarize_features(path)
    stage = result['stages']['msckf']
    assert stage['triangulation_failures'] == {'depth_below_min': 1}
    assert stage['first_update_frames'][0]['chi2_to_threshold']['max'] == 2.
    assert stage['raw_pixel_rms']['count'] == 2
    assert stage['raw_pixel_rms']['median'] == 2.
    assert abs(stage['raw_pixel_rms']['p90'] - 2.8) < 1e-12
    json.dumps(result, allow_nan=False)
