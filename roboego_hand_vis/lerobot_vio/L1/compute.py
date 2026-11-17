"""Prepare portable LeRobot episode inputs without changing source data."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def plan_session(session, input_mode='rectified', zupt=False):
    if input_mode not in ('raw', 'rectified'):
        raise ValueError('Invalid input mode')
    session = Path(session).resolve()
    episodes = sorted(p for p in (session / 'data').glob('chunk-*/episode_*') if p.is_dir())
    if not episodes:
        raise ValueError('No LeRobot episode directories found')
    plans = []
    for episode in episodes:
        prefix = f'{episode.parent.name}_{episode.name}_' if len(episodes) > 1 else ''
        calibration = session / 'calibrations'
        paths = dict(imu=episode / 'observation.imu.parquet',
                     manifest=calibration / f'{prefix}dataset_manifest.json',
                     rectify=calibration / f'{prefix}rectify_params.yaml',
                     camera_imu=calibration / 'camera_imu-camchain-imucam.yaml',
                     imu_calibration=calibration / 'camera_imu-imu.yaml',
                     stereo_calibration=calibration / 'camera-camchain.yaml')
        if input_mode == 'raw':
            for key in ('manifest', 'rectify', 'stereo_calibration'):
                paths.pop(key)
        for side in ('left', 'right'):
            key = f'observation.images.head_{side}' + ('_rectified' if input_mode == 'rectified' else '')
            paths[f'{side}_video'] = session / 'videos' / episode.parent.name / key / f'{episode.name}.mp4'
            paths[f'{side}_timestamps'] = episode / f'{key}.parquet'
        for key, path in paths.items():
            if not path.is_file():
                raise ValueError(f'Missing {key}: {path}; run L1 stereo rectification first')
        folder = ('vio_raw' if input_mode == 'raw' else 'vio') + ('_zupt' if zupt else '')
        plans.append(dict(session=session, episode=episode, input_mode=input_mode, **paths,
                          output=session / folder / episode.parent.name / episode.name))
    return plans


def validate_times(values, label, *, strictly_increasing=True):
    if not values or any(v is None or not math.isfinite(float(v)) for v in values):
        raise ValueError(f'{label}: empty or non-finite timestamps')
    values = [float(v) for v in values]
    if strictly_increasing and any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f'{label}: timestamps must strictly increase')
    return values


def normalize_imu(table):
    """Clean runtime input only; stable ordering keeps the first duplicate sample."""
    columns = ('timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az')
    if any(key not in table for key in columns):
        raise ValueError('IMU requires timestamp,gx,gy,gz,ax,ay,az columns')
    n = len(table['timestamp'])
    if not n or any(len(table[key]) != n for key in columns):
        raise ValueError('Empty or inconsistent IMU columns')
    if any(v is None or not math.isfinite(float(v)) for key in columns for v in table[key]):
        raise ValueError('Non-finite IMU timestamp or measurement')
    times = [float(t) for t in table['timestamp']]
    indices = []
    for index in sorted(range(n), key=lambda i: times[i]):
        if not indices or times[index] != times[indices[-1]]:
            indices.append(index)
    cleaned = {key: [float(table[key][i]) for i in indices] for key in columns}
    report = dict(policy='stable_sort_keep_first', input_samples=n, output_samples=len(indices),
                  duplicates_removed=n-len(indices),
                  backward_steps=sum(b < a for a,b in zip(times, times[1:])))
    return cleaned, report


def runtime_camera_samples(times):
    """Keep frame order and indices; skip timestamps the estimator cannot advance to."""
    from roboego_hand_vis.vio.session import CameraTimestamp
    samples = []
    skipped = []
    for index, timestamp in enumerate(times):
        if samples and timestamp <= samples[-1].timestamp:
            skipped.append(index)
        else:
            samples.append(CameraTimestamp(index, timestamp))
    return samples, dict(source_frames=len(times), retained_frames=len(samples),
                         skipped_frames=len(skipped), skipped_frame_indices=skipped,
                         policy='keep_source_order_skip_nonadvancing')


def configure_zupt(config_path, enabled):
    from roboego_hand_vis.vio.calibration import load_yaml_mapping, _write_opencv_yaml
    settings = load_yaml_mapping(config_path)
    settings['try_zupt'] = bool(enabled)
    _write_opencv_yaml(config_path, settings)
    return {key: value for key, value in settings.items() if key == 'try_zupt' or key.startswith('zupt_')}


def configure_initialization(config_path, mode='auto'):
    """Select an existing backend initializer without altering its thresholds."""
    from roboego_hand_vis.vio.calibration import load_yaml_mapping, _write_opencv_yaml
    if mode not in ('auto', 'static', 'dynamic'):
        raise ValueError('init_mode must be auto, static or dynamic')
    settings = load_yaml_mapping(config_path)
    settings['init_dyn_use'] = mode in ('auto', 'dynamic')
    if mode == 'dynamic':
        # OpenVINS gates the static initializer on a positive IMU threshold.
        settings['init_imu_thresh'] = 0.0
    _write_opencv_yaml(config_path, settings)
    return dict(mode=mode, **{k: v for k, v in settings.items() if k.startswith('init_')},
                try_zupt=bool(settings.get('try_zupt', False)),
                wait_for_jerk=not bool(settings.get('try_zupt', False)))


def prepare(plan, work, *, stereo_tolerance_s=.001, downsample_cameras=False, raw_center_crop_calibration=False, zupt=False):
    import cv2
    import pyarrow.parquet as pq
    import yaml
    from roboego_hand_vis.vio.calibration import prepare_rectified_openvins_calibration
    from roboego_hand_vis.vio.session import CameraTimestamp, IMU_FIELDS, match_stereo_timestamps

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if plan.get('camera_mode', 'stereo') != 'stereo':
        raise ValueError('Only stereo VIO is supported')
    if 'calibration_jsonl' in plan:
        from roboego_hand_vis.lerobot_vio.L3.compute import materialize_calibration
        plan = materialize_calibration(plan, work)
    times = {}
    if plan.get('input_mode', 'rectified') == 'raw':
        from .raw import prepare_raw
        calibration, videos = prepare_raw(plan, work, raw_center_crop_calibration, downsample_cameras)
        for side in ('left', 'right'):
            times[side] = validate_times(pq.read_table(plan[f'{side}_timestamps'])['timestamp'].to_pylist(), side, strictly_increasing=False)
            capture = cv2.VideoCapture(str(videos[side]))
            try:
                if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(times[side]):
                    raise ValueError(f'{side}: raw video/timestamp count mismatch')
            finally:
                capture.release()
    else:
        manifest = json.loads(plan['manifest'].read_text(encoding='utf-8'))
        for side in ('left', 'right'):
            table = pq.read_table(plan[f'{side}_timestamps']).to_pydict()
            times[side] = validate_times(table['timestamp'], side, strictly_increasing=False)
            video = plan[f'{side}_video']
            capture = cv2.VideoCapture(str(video))
            try:
                if not capture.isOpened():
                    raise ValueError(f'Cannot open video: {video}')
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            finally:
                capture.release()
            intrinsics = manifest['intrinsics'][side]
            video_meta = manifest['videos'][side]
            if count != len(times[side]) or size != (int(video_meta['width']), int(video_meta['height'])):
                raise ValueError(f'{side}: video frame count or resolution disagrees with timestamps/calibration')
            # The original VIO converter stores dimensions next to intrinsics.
            intrinsics.update(width=size[0], height=size[1])
            manifest['videos'][side]['path'] = str(video)
        if manifest.get('reference_camera', 'left') != 'left':
            raise ValueError('Only left-reference stereo is supported')
        runtime_manifest = work / 'dataset_manifest.json'
        runtime_manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        rectification = yaml.safe_load(plan['rectify'].read_text(encoding='utf-8'))
        runtime_rectify = work / 'rectify_params.yaml'
        runtime_rectify.write_text(yaml.safe_dump(dict(rectified=rectification.get('rectified', rectification))), encoding='utf-8')
        calibration = prepare_rectified_openvins_calibration(
            runtime_manifest, plan['camera_imu'], plan['imu_calibration'], runtime_rectify,
            work / 'config', stereo_camera_path=plan['stereo_calibration'],
            downsample_cameras=downsample_cameras)
        videos = {side: plan[f'{side}_video'] for side in ('left', 'right')}
    zupt_settings = configure_zupt(calibration.estimator_config_path, zupt)
    camera_samples = {side: runtime_camera_samples(times[side]) for side in ('left', 'right')}
    (work / 'camera_timestamp_handling.json').write_text(
        json.dumps({side: camera_samples[side][1] for side in camera_samples}, indent=2), encoding='utf-8')
    matches = match_stereo_timestamps(
        camera_samples['left'][0], camera_samples['right'][0], tolerance_s=stereo_tolerance_s)
    pairs = work / 'stereo_pairs.csv'
    with pairs.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['timestamp', 'left_frame_index', 'right_frame_index'])
        writer.writerows((p.timestamp, p.left_frame_index, p.right_frame_index) for p in matches.pairs)
    imu = pq.read_table(plan['imu']).to_pydict()
    imu, normalization = normalize_imu(imu)
    imu_times = validate_times(imu['timestamp'], 'IMU')
    columns = ['gx', 'gy', 'gz', 'ax', 'ay', 'az']
    if any(key not in imu or len(imu[key]) != len(imu_times) for key in columns):
        raise ValueError('IMU requires timestamp,gx,gy,gz,ax,ay,az columns')
    from .imu_diagnostics import summarize
    diagnostics = summarize(imu, times['left'], calibration.timeshift_cam_imu_s)
    diagnostics['normalization'] = normalization
    (work / 'imu_diagnostics.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding='utf-8')
    imu_path = work / 'imu_clean.csv'
    with imu_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(IMU_FIELDS)
        for index, timestamp in enumerate(imu_times):
            values = [imu[key][index] for key in columns]
            if any(v is None or not math.isfinite(float(v)) for v in values):
                raise ValueError('Non-finite IMU measurement')
            writer.writerow([timestamp, *values])
    return dict(config=calibration.estimator_config_path, stereo_pairs=pairs, imu=imu_path,
                imu_diagnostics=diagnostics, videos=videos, zupt_settings=zupt_settings)


def run_episode(plan, *, openvins_root, start_s=0., duration_s=None,
                stereo_tolerance_s=.001, downsample_cameras=False,
                time_offset_mode='estimate', timing_bootstrap='auto', prepare_only=False, quality_thresholds=None,
                raw_center_crop_calibration=False, zupt=False, init_mode='auto'):
    root = Path(openvins_root).expanduser().resolve()
    entry = root / 'ov_offline/run.py'
    binary = root / 'ov_msckf/build-release/openvins_offline_native'
    if not prepare_only and (not entry.is_file() or not binary.is_file()):
        raise ValueError(f'Build-release the ROS-free offline target first: {binary}')
    output = plan['output']
    if not output.resolve().is_relative_to(plan['session']) or output.is_symlink():
        raise ValueError('VIO output escapes dataset')
    # A fresh run preserves previous results and avoids mixing old/new artifacts.
    work = output / '.work' / uuid4().hex
    work.mkdir(parents=True, exist_ok=False)
    inputs = prepare(plan, work, stereo_tolerance_s=stereo_tolerance_s,
                     downsample_cameras=downsample_cameras, raw_center_crop_calibration=raw_center_crop_calibration, zupt=zupt)
    initialization = configure_initialization(inputs['config'], init_mode)
    (work / 'initialization_settings.json').write_text(
        json.dumps(initialization, indent=2, allow_nan=False), encoding='utf-8')
    if prepare_only:
        return dict(status='prepared', output=str(work))
    destination = work / 'result'
    videos = inputs.get('videos', {side: plan[f'{side}_video'] for side in ('left', 'right')})
    command = [sys.executable, str(entry), '--binary', str(binary),
               '--config', str(inputs['config']), '--left_video', str(videos['left']),
               '--right_video', str(videos['right']), '--stereo_pairs', str(inputs['stereo_pairs']),
               '--imu', str(inputs['imu']), '--output_dir', str(destination), '--start_s', str(start_s),
               '--time-offset-mode', time_offset_mode, '--timing-bootstrap', timing_bootstrap]
    if duration_s is not None:
        command.extend(['--duration_s', str(duration_s)])
    with (work / 'console.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'OpenVINS exited {result.returncode}; see {work / "console.log"}')
    import numpy as np
    trajectory = destination / 'trajectory/trajectory.npz'
    with np.load(trajectory, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    import pyarrow.parquet as pq
    from .quality import check_trajectory
    try:
        quality = check_trajectory(arrays, pq.read_table(plan['left_timestamps'])['timestamp'].to_pylist(),
                                   quality_thresholds, camera_mode=plan.get('camera_mode', 'stereo'))
    except (ValueError, KeyError) as exc:
        (work / 'quality.json').write_text(json.dumps(dict(status='invalid_output', error=str(exc))), encoding='utf-8')
        raise ValueError(f'{exc}; diagnostics: {work}') from exc
    (work / 'quality.json').write_text(json.dumps(quality, indent=2, allow_nan=False), encoding='utf-8')
    report = json.loads((destination / 'trajectory/run_report.json').read_text())
    if report.get('status') != 'complete' or not report.get('initialized'):
        raise ValueError('OpenVINS report does not indicate successful initialization')
    report.update(source_session=str(plan['session']), source_episode=str(plan['episode']),
                  input_mode=plan.get('input_mode', 'rectified'),
                  timestamp_source='LeRobot sensor parquet timestamp, seconds', quality_status=quality['status'], quality=quality)
    if 'calibration_jsonl' in plan:
        report['calibration_source'] = json.loads((work / 'source_calibration/source.json').read_text(encoding='utf-8'))
        report['source_imu'] = plan['source_imu']
    from .imu_diagnostics import compare
    comparison = compare(pq.read_table(plan['imu']).to_pydict(), arrays, quality)
    comparison['time_offset'] = {key: report.get('offline', {}).get(key) for key in
                                 ('initial_time_offset_s', 'final_time_offset_s', 'time_offset_estimated')}
    initial, final = comparison['time_offset']['initial_time_offset_s'], comparison['time_offset']['final_time_offset_s']
    comparison['time_offset']['change_s'] = final-initial if initial is not None and final is not None else None
    report['camera_mode'] = plan.get('camera_mode', 'stereo')
    report['imu_diagnostics'] = inputs.get('imu_diagnostics')
    report['zupt_settings'] = inputs.get('zupt_settings', {'try_zupt': zupt})
    report['initialization_settings'] = initialization
    report['imu_vio_comparison'] = comparison
    (work / 'imu_vio_comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding='utf-8')
    report['artifacts']['trajectory_npz'] = 'trajectory.npz'
    # Publish the pair together using the existing reversible file publisher.
    from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import publish_files
    staging = plan['session'] / f'.vio-publish-{uuid4().hex}'
    staging.mkdir()
    from .storage import stage_features
    comparison_only = plan.get('input_mode') == 'raw' 
    metadata = (dict(published_to_lerobot=False, reason='Comparison preserves canonical derived features',
                     source_camera=plan.get('source_camera', 'observation.images.head_left')) if comparison_only
                else stage_features(plan, arrays, staging, quality))
    if not comparison_only:
        metadata.update(published_to_lerobot=True, zupt_enabled=bool(zupt))
        provenance = staging / plan['episode'].relative_to(plan['session']) / (
            'left_camera_pose_provenance.json' if plan.get('publish_left_camera_pose') else 'relative_head_pose_provenance.json')
        provenance.write_text(json.dumps(dict(**metadata, report=str((output / 'report.json').relative_to(plan['session'])),
                                              zupt_settings=report['zupt_settings']), indent=2, allow_nan=False), encoding='utf-8')
    relative_output = output.relative_to(plan['session'])
    staged_output = staging / relative_output
    staged_output.mkdir(parents=True)
    arrays['quality_status'] = np.array(quality['status'])
    np.savez_compressed(staged_output / 'trajectory.npz', **arrays)
    report['lerobot'] = metadata
    estimator_csv = destination / 'trajectory/estimator_diagnostics.csv'
    if estimator_csv.is_file():
        import shutil
        from .estimator_diagnostics import summarize as summarize_estimator
        estimator_summary = summarize_estimator(estimator_csv)
        report['estimator_diagnostics'] = estimator_summary
        feature_csv = destination / 'trajectory/feature_diagnostics.csv'
        if feature_csv.is_file():
            from .estimator_diagnostics import summarize_features
            estimator_summary['feature_details'] = summarize_features(feature_csv)
            shutil.copy2(feature_csv, staged_output / 'feature_diagnostics.csv')
        shutil.copy2(estimator_csv, staged_output / 'estimator_diagnostics.csv')
        (staged_output / 'estimator_diagnostics.json').write_text(json.dumps(estimator_summary, indent=2, allow_nan=False), encoding='utf-8')
    (staged_output / 'imu_vio_comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding='utf-8')
    (staged_output / 'imu_diagnostics.json').write_text(json.dumps(inputs.get('imu_diagnostics'), indent=2, allow_nan=False), encoding='utf-8')
    (staged_output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    publish_files(staging, plan['session'])
    return dict(status='failed_quality' if quality['status'] == 'failed' else 'done',
                quality=quality, imu_vio_status=comparison['status'],
                output=str(output), log=str(work / 'console.log'))


def run_initialization_policy(plan, options):
    options = dict(options)
    policy = options.pop('init_mode')
    if options.pop('zupt_fallback', False) or options.pop('zupt', False):
        raise ValueError('Use init-mode alone for the initialization policy')
    modes = {
        'static-zupt': ('static-zupt',),
        'dynamic': ('dynamic',),
        'static-then-dynamic': ('static-zupt', 'dynamic'),
        'dynamic-then-static': ('dynamic', 'static-zupt'),
    }[policy]
    attempts = []
    base = 'vio_raw' if plan.get('input_mode') == 'raw' else 'vio'
    for mode in modes:
        attempt = dict(plan)
        attempt['output'] = plan['session'] / (base + ('_zupt' if mode == 'static-zupt' else '')) / plan['episode'].parent.name / plan['episode'].name
        try:
            result = run_episode(attempt, **options, init_mode='static' if mode == 'static-zupt' else 'dynamic',
                                 zupt=mode == 'static-zupt')
        except (RuntimeError, ValueError) as exc:
            # Invalid input and publication errors must remain visible, without retry.
            if not (str(exc).startswith('OpenVINS exited ') or
                    str(exc).startswith('OpenVINS report does not indicate successful initialization') or
                    str(exc).startswith('No valid poses; diagnostics:')):
                raise
            result = dict(status='failed', error=str(exc), output=str(attempt['output']))
        attempts.append(dict(mode=mode, **result))
        if result['status'] not in ('failed', 'failed_quality') or options.get('prepare_only'):
            break
    selected = attempts[-1]['mode'] if result['status'] == 'done' else None
    selection = dict(policy=policy, selected_mode=selected, attempts=attempts)
    path = plan['session'] / base / plan['episode'].parent.name / plan['episode'].name / 'fallback_selection.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    temporary = path.with_name(f'.{path.name}-{uuid4().hex}')
    temporary.write_text(json.dumps(selection, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)
    return dict(**result, selected_mode=selected, attempted_modes=[a['mode'] for a in attempts], selection_report=str(path))


def run_with_policy(plan, options):
    options = dict(options)
    mode = options.get('init_mode', 'auto')
    if mode in ('static-zupt', 'dynamic', 'static-then-dynamic', 'dynamic-then-static'):
        return run_initialization_policy(plan, options)
    fallback = options.pop('zupt_fallback', False)
    if not fallback:
        return run_episode(plan, **options)
    if options.get('zupt'):
        raise ValueError('zupt and zupt_fallback are mutually exclusive')
    attempts = []
    try:
        result = run_episode(plan, **options)
    except RuntimeError as exc:
        # Input/structural/publication errors are not reasons to change the estimator.
        if not str(exc).startswith('OpenVINS exited '):
            raise
        result = dict(status='failed', error=str(exc), output=str(plan['output']))
    attempts.append(dict(mode='normal', **result))
    if result['status'] in ('failed_quality', 'failed') and not options.get('prepare_only'):
        retry_plan = dict(plan)
        base = plan.get('output_family') or ('vio_raw' if plan.get('input_mode') == 'raw' else 'vio')
        retry_plan['output'] = plan['session'] / (base + '_zupt') / plan['episode'].parent.name / plan['episode'].name
        try:
            result = run_episode(retry_plan, **{**options, 'zupt': True})
        except Exception as exc:
            result = dict(status='failed', error=str(exc), output=str(retry_plan['output']))
        attempts.append(dict(mode='zupt', **result))
    selected = attempts[-1]['mode'] if result['status'] == 'done' else None
    selection = dict(policy='normal_then_zupt', selected_mode=selected, attempts=attempts)
    # Retain the decision beside the normal-run report even when retry fails.
    path = plan['output'] / 'fallback_selection.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    temporary = path.with_name(f'.{path.name}-{uuid4().hex}')
    temporary.write_text(json.dumps(selection, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)
    return dict(**result, selected_mode=selected, attempted_modes=[a['mode'] for a in attempts], selection_report=str(path))


def run_session(plans, options):
    """Serialize episodes sharing meta/info.json; parallelize across datasets."""
    results = []
    for plan in plans:
        try:
            result = run_with_policy(plan, options)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result = dict(status='failed', error=str(exc))
        results.append(dict(episode=str(plan['episode']), **result))
    return results
