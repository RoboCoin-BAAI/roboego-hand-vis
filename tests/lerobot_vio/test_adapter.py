import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roboego_hand_vis.lerobot_vio.L1 import compute
from roboego_hand_vis.lerobot_headband_integrity.L1.run import discover


def test_batch_names_and_multiple_episodes(tmp_path):
    session = tmp_path / 'batch_000001' / 'session'
    (session / 'meta').mkdir(parents=True)
    (session / 'meta/info.json').write_text('{}')
    for index in range(2):
        episode = session / 'data/chunk-000' / f'episode_{index:06d}'
        episode.mkdir(parents=True)
    # Plan errors identify missing inputs instead of silently omitting an episode.
    assert discover(tmp_path) == [session]
    assert discover(session) == [session]
    with pytest.raises(ValueError, match='Missing imu'):
        compute.plan_session(session)


@pytest.mark.parametrize('values', [[], [0., 0.], [1., 0.], [float('nan')], [None]])
def test_reject_invalid_sensor_times(values):
    with pytest.raises(ValueError):
        compute.validate_times(values, 'sensor')


@pytest.mark.parametrize('exit_code, quality_limits', [(0, None), (3, None), (0, {'min_valid_ratio': 1., 'max_distance_m': 0.})])
@pytest.mark.parametrize('zupt', [False, True])
@pytest.mark.parametrize('init_mode', ['auto', 'static'])
def test_publish_only_successful_run(tmp_path, monkeypatch, exit_code, quality_limits, zupt, init_mode):
    session = tmp_path / 'session'
    output = session / ('vio_zupt' if zupt else 'vio') / 'chunk-000/episode_000000'
    output.mkdir(parents=True)
    (output / 'trajectory.npz').write_bytes(b'previous result')
    root = tmp_path / 'open_vins'
    for relative in ('ov_offline/run.py', 'ov_msckf/build-release/openvins_offline_native'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    plan = dict(session=session, episode=session / 'data/chunk-000/episode_000000',
                output=output, left_video=tmp_path / 'left.mp4', right_video=tmp_path / 'right.mp4',
                left_timestamps=session / 'data/chunk-000/episode_000000/left.parquet')
    import pyarrow as pa
    import pyarrow.parquet as pq
    plan['episode'].mkdir(parents=True)
    pq.write_table(pa.table(dict(timestamp=[0., 1.], episode_index=[0, 0])), plan['left_timestamps'])
    plan['imu'] = plan['episode'] / 'observation.imu.parquet'
    pq.write_table(pa.table(dict(timestamp=[0., .5, 1.], ax=[0.]*3, ay=[0.]*3, az=[9.81]*3,
                                gx=[0.]*3, gy=[0.]*3, gz=[0.]*3)), plan['imu'])
    (session / 'meta').mkdir()
    (session / 'meta/info.json').write_text(json.dumps(dict(fps=30., features={})))
    from roboego_hand_vis.vio.calibration import _write_opencv_yaml, _estimator_config, load_yaml_mapping
    config = tmp_path / 'estimator.yaml'
    settings = _estimator_config(tmp_path, downsample_cameras=False)
    settings['try_zupt'] = zupt
    _write_opencv_yaml(config, settings)
    monkeypatch.setattr(compute, 'prepare', lambda *a, **kw: dict(config=config, stereo_pairs='pairs', imu='imu'))

    def backend(command, **kwargs):
        configured = load_yaml_mapping(Path(command[command.index('--config') + 1]))
        assert configured['init_dyn_use'] is (init_mode == 'auto')
        assert configured['try_zupt'] is zupt
        dest = Path(command[command.index('--output_dir') + 1]) / 'trajectory'
        dest.mkdir(parents=True)
        from test_quality import trajectory
        arrays = trajectory()
        arrays['T_world_imu'][1, 0, 3] = .1
        arrays['T_world_camera'][1, 0, 3] = .1
        np.savez(dest / 'trajectory.npz', **arrays)
        (dest / 'run_report.json').write_text(json.dumps(dict(status='complete', initialized=True, artifacts={})))
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(compute.subprocess, 'run', backend)
    if exit_code:
        with pytest.raises(RuntimeError, match='exited 3'):
            compute.run_episode(plan, openvins_root=root, zupt=zupt, init_mode=init_mode)
        assert (output / 'trajectory.npz').read_bytes() == b'previous result'
    else:
        result = compute.run_episode(plan, openvins_root=root, quality_thresholds=quality_limits, zupt=zupt, init_mode=init_mode)
        saved_report = json.loads((output / 'report.json').read_text())
        assert saved_report['initialization_settings']['mode'] == init_mode
        assert saved_report['initialization_settings']['wait_for_jerk'] is (not zupt)
        assert result['status'] == ('failed_quality' if quality_limits else 'done')
        assert json.loads((output / 'report.json').read_text())['artifacts']['trajectory_npz'] == 'trajectory.npz'
        feature = pq.read_table(plan['episode'] / 'observation.relative_head_pose.parquet').to_pydict()
        assert feature['timestamp'] == [0., 1.]
        assert feature['qw'] == [1., 1.]
        assert feature['x'] == [0., .1]
        provenance = json.loads((plan['episode'] / 'relative_head_pose_provenance.json').read_text())
        assert provenance['zupt_enabled'] is zupt
        assert provenance['published_to_lerobot'] is True
        assert provenance['report'] == (output / 'report.json').relative_to(session).as_posix() or provenance['report'] == str((output / 'report.json').relative_to(session))
        stats = json.loads((session / 'meta/episodes_stats.jsonl').read_text())
        assert stats['stats']['observation.relative_head_pose']['count'] == [2]
