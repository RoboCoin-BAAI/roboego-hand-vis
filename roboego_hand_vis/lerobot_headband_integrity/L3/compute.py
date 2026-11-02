"""L3 checks with missing-calibration recovery from matching devices."""
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
from uuid import uuid4
import wave

from ..common import Context, finite, guarded, result, valid_times, stereo_boundary_checks
from ..L1.imu import signal_quality
from ..camera_calibration import camera_error


INDEX_COLUMNS = {'timestamp', 'episode_index', 'frame_index', 'index', 'task_index'}
CAMERAS = ('left_wrist_rgb', 'right_wrist_rgb', 'third_rgb',
           'headband_bright_rgb', 'headband_left_rgb', 'headband_right_rgb',
           'headband_bleft_rgb', 'eye_rgb')
HEADBAND_CAMERAS = ('headband_bright_rgb', 'headband_left_rgb', 'headband_right_rgb',
                    'headband_bleft_rgb')


def transform_valid(matrix):
    if not isinstance(matrix, list) or len(matrix) != 4:
        return False
    if any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        return False
    if not all(number(v) for row in matrix for v in row):
        return False
    if any(abs(a-b) > 1e-6 for a,b in zip(matrix[3], [0,0,0,1])):
        return False
    r = matrix
    if any(abs(sum(r[k][i]*r[k][j] for k in range(3)) - (i == j)) > 1e-5
           for i in range(3) for j in range(3)):
        return False
    det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
           -r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
           +r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    return abs(det-1) <= 1e-5


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_feature(ctx, episode, feature, declaration, master):
    dtype = declaration.get('dtype')
    if dtype not in ('video', 'audio') and not feature.startswith('observation.headband_imu'):
        return []
    table = ctx.table(episode, feature)
    ts = table.get('timestamp', [])
    finite_ts = bool(ts) and finite(ts)
    is_imu = feature.startswith('observation.headband_imu')
    checks = [result('timestamps', valid_times(ts), samples=len(ts),
                     level='warning' if (is_imu or dtype == 'video') and finite_ts else 'fail')]
    if dtype not in ('video', 'audio'):
        values = {k: v for k, v in table.items() if k not in INDEX_COLUMNS}
        width = math.prod(declaration['shape'])
        checks.append(result('width', len(values) == width, expected=width, actual=len(values)))
        names = declaration.get('names')
        if names:
            checks.append(result('columns', set(names) == set(values)))
        checks.append(result('finite_values', bool(values) and all(finite(v) for v in values.values())))
        checks.append(result('all_zero', any(x is not None and x != 0 for v in values.values() for x in v),
                             level='warning'))
        if values and all(finite(v) for v in values.values()):
            stats = signal_quality(values)
            checks.append(result('frozen', not stats['frozen_columns'],
                                 level='fail', **stats))
            checks.append(result('jump', not stats['jump_columns'], **stats))
    if dtype == 'video':
        camera = feature.removeprefix('observation.images.')
        run = subprocess.run([ctx.ffprobe, '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate',
            '-of', 'json', str(ctx.video(episode, camera))], capture_output=True, text=True, check=True)
        stream, = json.loads(run.stdout)['streams']
        frames = int(stream['nb_read_frames'])
        checks.extend([
            result('video/decodable', frames > 0 and not run.stderr.strip(), detail=run.stderr.strip()),
            result('video/timestamp_count', frames == len(next(iter(table.values()))),
                   frames=frames, timestamps=len(next(iter(table.values())))),
            result('video/dimensions', [stream['height'], stream['width'], 3] == declaration['shape']),
            result('video/fps', abs(float(Fraction(stream['avg_frame_rate'])) - ctx.fps(camera)) <= 1,
                   level='warning'),
        ])
    elif dtype == 'audio':
        path = ctx.root / 'audios' / episode.parent.name / feature / f'{episode.name}.wav'
        with wave.open(str(path), 'rb') as wav:
            declared = wav.getnframes()
            frame_size = wav.getnchannels() * wav.getsampwidth()
            actual = 0
            while block := wav.readframes(65536):
                actual += len(block)
            checks.extend([
                result('audio/readable', actual == declared * frame_size and declared > 0),
                result('audio/sample_rate', wav.getframerate() == declaration['sample_rate']),
                result('audio/sample_count', sum(table['num_samples']) == declared,
                       wav_samples=declared, parquet_samples=sum(table['num_samples'])),
            ])
            if master and finite(master):
                duration = declared / wav.getframerate()
                expected = max(master) - min(master) + 1 / float(ctx.info['fps'])
                checks.append(result('audio/duration', abs(duration - expected) <= ctx.duration_tolerance_s,
                                     duration_s=duration, master_duration_s=expected))
    return checks


def check_calibration(ctx):
    path = ctx.root / 'meta/calibrations.jsonl'
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    checks = [result('episodes', sorted(r['episode_index'] for r in rows) ==
                     sorted(int(p.name.removeprefix('episode_')) for p in ctx.episodes))]
    for row in rows:
        prefix = str(row['episode_index'])
        head = row.get('calibrations', {}).get('ego_headband', {})
        for key in ('camera-camchain', 'camera_imu-camchain-imucam', 'camera_imu-imu'):
            checks.append(result(f'{prefix}/{key}/present', isinstance(head.get(key), dict) and bool(head[key])))
        imu = head.get('camera_imu-imu', {}).get('imu0', {})
        for field in ('accelerometer_noise_density', 'accelerometer_random_walk',
                      'gyroscope_noise_density', 'gyroscope_random_walk', 'update_rate'):
            value = imu.get(field) if isinstance(imu, dict) else None
            checks.append(result(f'{prefix}/camera_imu-imu/imu0/{field}', number(value) and value > 0))
        for key in ('camera-camchain', 'camera_imu-camchain-imucam'):
            chain = head.get(key, {})
            checks.append(result(f'{prefix}/{key}/cameras', all(f'cam{i}' in chain for i in range(4))))
            for name, cam in chain.items():
                error = camera_error(cam)
                checks.append(result(f'{prefix}/{key}/{name}/intrinsics', error is None,
                                     detail=error, camera_model=cam.get('camera_model', 'pinhole'),
                                     distortion_model=cam.get('distortion_model', 'equidistant')))
                if key == 'camera_imu-camchain-imucam':
                    checks.append(result(f'{prefix}/{key}/{name}/timeshift_cam_imu',
                                         number(cam.get('timeshift_cam_imu'))))
                for field in ('T_cam_imu', 'T_cn_cnm1'):
                    required = ((field == 'T_cam_imu' and key == 'camera_imu-camchain-imucam')
                                or (field == 'T_cn_cnm1' and name in ('cam1', 'cam2', 'cam3')))
                    if field in cam or required:
                        matrix = cam.get(field, [])
                        valid = transform_valid(matrix)
                        checks.append(result(f'{prefix}/{key}/{name}/{field}', valid))
    return checks


def build_report(session, calibration_donors=(), **options):
    checks = []
    repair = None
    try:
        ctx = Context(Path(session), **options)
        features = ctx.info['features']
        streams = {k: v for k, v in features.items() if k.startswith('observation.') or k == 'action'}
        imu_feature = 'observation.headband_imu0'
        checks.append(result('dataset/imu_feature',
                             features.get(imu_feature, {}).get('dtype') in ('float32', 'float64')
                             and features[imu_feature].get('shape') == [6]))
        for episode in ctx.episodes:
            checks.append(result(f'{episode.parent.name}/{episode.name}/{imu_feature}/present',
                                 (episode / f'{imu_feature}.parquet').is_file()))
        headband_present = True
        for episode in ctx.episodes:
            for camera in HEADBAND_CAMERAS:
                feature = f'observation.images.{camera}'
                present = (features.get(feature, {}).get('dtype') == 'video'
                           and ctx.video(episode, camera).is_file()
                           and (episode / f'{feature}.parquet').is_file())
                headband_present &= present
                checks.append(result(f'headband/{episode.name}/{camera}/present', present))
        if headband_present:
            from .calibration_repair import repair_missing
            try:
                repair = repair_missing(ctx, calibration_donors)
            except Exception as exc:
                repair = dict(status='error', detail=f'{type(exc).__name__}: {exc}')
        for item in guarded('readable', lambda: check_calibration(ctx)):
            item['name'] = 'calibration/' + item['name']
            checks.append(item)
        checks.append(result('dataset/audio_feature',
                             features.get('observation.headband_microphone', {}).get('dtype') == 'audio'))
        checks.append(result('dataset/features', all(f'observation.images.{c}' in streams for c in CAMERAS)))
        checks.append(result('dataset/episode_count', ctx.info['total_episodes'] == len(ctx.episodes)))
        total = 0
        for episode in ctx.episodes:
            prefix = f'{episode.parent.name}/{episode.name}'
            boundaries = guarded('readable', lambda: stereo_boundary_checks(
                ctx, episode, 'headband_left_rgb', 'headband_right_rgb'))
            for item in boundaries:
                item['name'] = f'{prefix}/stereo_boundary/{item["name"]}'
            checks.extend(boundaries)
            try:
                master = ctx.table(episode, 'master_index')['timestamp']
            except Exception as exc:
                checks.append(result(f'{prefix}/master_index/readable', False, detail=str(exc)))
                master = []
            checks.append(result(f'{prefix}/master_index/timestamps', valid_times(master),
                                 level='warning' if master and finite(master) else 'fail'))
            total += len(master)
            # Native-rate streams need not have the same sample count as the master.
            for feature, declaration in streams.items():
                items = guarded('readable', lambda: check_feature(ctx, episode, feature, declaration, master))
                for item in items:
                    item['name'] = f'{prefix}/{feature}/{item["name"]}'
                checks.extend(items)
        checks.append(result('dataset/total_frames', total == ctx.info['total_frames'], actual=total))
    except Exception as exc:
        checks.append(result('dataset/readable', False, detail=f'{type(exc).__name__}: {exc}'))
    failed = [c['name'] for c in checks if not c['passed'] and c['level'] == 'fail']
    warnings = [c['name'] for c in checks if not c['passed'] and c['level'] == 'warning']
    return dict(schema_version='1.0', stage='lerobot_headband_integrity_l3',
                source_session=str(Path(session).resolve()), created_at=datetime.now(timezone.utc).isoformat(),
                passed=not failed, gate=dict(passed=not failed, failed_checks=failed, warnings=warnings),
                calibration_repair=repair, checks=checks)


def export_report(session, **options):
    report = build_report(session, **options)
    path = Path(session) / 'annotation/quality_report.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return dict(session=str(session), report=str(path), **report['gate'])
