"""Repair only IMU and camera-IMU calibration from same-device QC-passed donors."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from uuid import uuid4

import yaml
from ..camera_calibration import camera_error

PATTERN = re.compile(r'^(cp_\d+)_session_(\d{8}_\d{6})$')
FILES = ('camera_imu-imu.yaml', 'camera_imu-camchain-imucam.yaml')


def complete_vio_file(path):
    """Check required VIO fields without comparing calibration to video size."""
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            raise ValueError('not a YAML mapping')
        if path.name == FILES[0]:
            imu = data['imu0']
            for key in ('accelerometer_noise_density', 'accelerometer_random_walk',
                        'gyroscope_noise_density', 'gyroscope_random_walk', 'update_rate'):
                value = imu[key]
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'invalid imu0.{key}')
        else:
            for name in ('cam0', 'cam1'):
                entry = data[name]
                error = camera_error(entry)
                if error:
                    raise ValueError(f'{name}: {error}')
                if not isinstance(entry['timeshift_cam_imu'], (int, float)) or not math.isfinite(entry['timeshift_cam_imu']):
                    raise ValueError(f'invalid {name} timeshift')
                matrix = entry['T_cam_imu']
                if len(matrix) != 4 or any(len(row) != 4 for row in matrix) or not all(isinstance(v, (int, float)) and math.isfinite(v) for row in matrix for v in row):
                    raise ValueError(f'invalid {name} transform')
                if any(abs(a-b) > 1e-6 for a,b in zip(matrix[3], (0,0,0,1))):
                    raise ValueError(f'invalid {name} homogeneous row')
                if any(abs(sum(matrix[k][i]*matrix[k][j] for k in range(3))-(i==j)) > 1e-5 for i in range(3) for j in range(3)):
                    raise ValueError(f'invalid {name} rotation')
                r = matrix
                det = r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0])
                if abs(det-1) > 1e-5:
                    raise ValueError(f'invalid {name} rotation determinant')
        return True, None
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as exc:
        return False, str(exc)


def donors(root):
    result = []
    reports = set(root.glob('batch_*/*/annotation/quality_report.json')) | set(root.glob('*/annotation/quality_report.json'))
    for report in sorted(reports):
        session = report.parent.parent.resolve()
        if not session.is_relative_to(root) or not PATTERN.fullmatch(session.name):
            continue
        try:
            data = json.loads(report.read_text(encoding='utf-8'))
            if data.get('passed') is not True or data.get('source_session') != str(session):
                continue
            paths = [session / 'calibrations' / name for name in FILES]
            if any(p.is_symlink() or not p.resolve().is_relative_to(session) or not complete_vio_file(p)[0] for p in paths):
                continue
            result.append((session, {p.name: p.read_bytes() for p in paths}))
        except (OSError, ValueError):
            continue
    return result


def cached_donors(root, cache):
    import base64
    import fcntl
    cache=Path(cache)
    cache.parent.mkdir(parents=True,exist_ok=True)
    print(f'[calibration-donors] waiting for shared snapshot: {cache}',flush=True)
    with cache.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not cache.exists():
            print('[calibration-donors] building snapshot once (shallow directory scan)',flush=True)
            items=donors(root)
            data=dict(root=str(root),items=[dict(session=str(session),files={k:base64.b64encode(v).decode('ascii')
                      for k,v in files.items()}) for session,files in items])
            tmp=cache.with_name('.'+cache.name+'-'+uuid4().hex)
            tmp.write_text(json.dumps(data),encoding='utf-8')
            tmp.replace(cache)
            print(f'[calibration-donors] ready: {len(items)} donors',flush=True)
        data=json.loads(cache.read_text(encoding='utf-8'))
        if data['root']!=str(root):
            raise ValueError('Calibration donor cache root mismatch')
        return [(Path(item['session']),{k:base64.b64decode(v) for k,v in item['files'].items()}) for item in data['items']]


def repair_sessions(sessions, root, donor_cache=None):
    root = Path(root).resolve()
    candidates = None
    outcomes = {}
    for session in sessions:
        session = Path(session).resolve()
        if not session.is_relative_to(root):
            raise ValueError(f'Session outside repair root: {session}')
        invalid = {}
        for name in FILES:
            valid, reason = complete_vio_file(session / 'calibrations' / name)
            if not valid:
                invalid[name] = reason
        if not invalid:
            continue
        if candidates is None:
            candidates = cached_donors(root, donor_cache) if donor_cache else donors(root)
        identity = PATTERN.fullmatch(session.name)
        options = [(s, payload) for s,payload in candidates
                   if identity and s != session and PATTERN.fullmatch(s.name)[1] == identity[1]]
        record = dict(applied=False, original_errors=invalid)
        if not options:
            record['reason'] = 'no QC-passed donor with complete IMU and camera-IMU calibration for same cp ID'
        else:
            def distance(item):
                timestamp = datetime.strptime(PATTERN.fullmatch(item[0].name)[2], '%Y%m%d_%H%M%S')
                current = datetime.strptime(identity[2], '%Y%m%d_%H%M%S')
                return abs((timestamp-current).total_seconds()), str(item[0])
            donor, payloads = min(options, key=distance)
            copied = {}
            for name in invalid:
                target = session / 'calibrations' / name
                if target.is_symlink() or not target.resolve().is_relative_to(session):
                    raise ValueError(f'Unsafe calibration repair path: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                backup = None
                if target.exists():
                    backup = target.with_name(target.name + '.before_repair-' + uuid4().hex)
                    backup.write_bytes(target.read_bytes())
                payload = payloads[name]
                temporary = target.with_name('.camera-repair-' + uuid4().hex)
                temporary.write_bytes(payload)
                temporary.replace(target)
                copied[name] = dict(source_file=str(donor / 'calibrations' / name),
                                    sha256=hashlib.sha256(payload).hexdigest(), backup=str(backup) if backup else None)
            record.update(applied=True, source_session=str(donor), files=copied,
                          device_id=identity[1], copied_at=datetime.now().astimezone().isoformat())
        outcomes[str(session)] = record
        annotation = session / 'annotation'
        annotation.mkdir(parents=True, exist_ok=True)
        receipt = annotation / 'camera_calibration_repair.json'
        temporary = receipt.with_name('.camera-repair-receipt-' + uuid4().hex)
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(receipt)
    return outcomes
