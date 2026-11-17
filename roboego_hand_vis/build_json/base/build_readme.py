"""Build compact LeRobot meta/readme.json files without model calls.

Usage: python build_readme.py --root SESSION_OR_BATCH_OR_DATASET
size_bytes/file_count include all source files except meta/readme.json and its
temporary files, so repeated generation produces stable statistics.
"""
import argparse
import json
import math
import os
import re
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_samples import video_sample_count
from readme_schema import lowercase_values, normalize_modalities

import pyarrow.parquet as pq


def load(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()] if path.exists() else []


def infer_scene_type(category, area, environment=''):
    """Infer typical indoor/outdoor context; ambiguous V1 areas stay unknown."""
    environment = str(environment).lower()
    if environment.startswith(('outdoor_', 'street_', 'garden_')):
        return 'outdoor'
    if environment.startswith(('office_', 'kitchen_', 'bedroom_', 'bathroom_',
                               'living_room_', 'warehouse_', 'factory_', 'retail_')):
        return 'indoor'
    if category in ('other', 'unknown'):
        return 'unknown'
    if area in {'street', 'garden', 'farm'}:
        return 'outdoor'
    if area in {
        'kitchen', 'dining_area', 'living_room', 'bedroom', 'bathroom', 'household_storage',
        'retail_store', 'restaurant', 'cafe', 'hotel', 'hair_salon', 'beauty_salon',
        'warehouse', 'sorting_area', 'packing_area', 'shelving_area',
        'assembly', 'machining', 'inspection', 'maintenance', 'workshop',
        'office', 'laboratory', 'classroom', 'medical_area', 'care_area', 'studio',
        'gaming_area',
    }:
        return 'indoor'
    # A broad category alone cannot place delivery, construction, transport,
    # sports, entertainment, gatherings or pet services indoors or outdoors.
    if category == 'household':
        return 'indoor'
    # Compatibility with older annotations.
    if category == 'outdoor':
        return 'outdoor'
    if category in ('kitchen', 'living_room', 'bedroom', 'bathroom', 'office',
                    'warehouse', 'factory', 'retail_store'):
        return 'indoor'
    return 'unknown'


def depth_streams(session):
    """Read stored depth streams from legacy and multi-episode reports."""
    for path in sorted((session/'depth').glob('chunk-*/*/lerobot_depth.json')):
        report = load(path)
        reports = report.get('episodes', {}).values() if report.get('schema_version') == 2 else [report]
        for report in reports:
            storage = report.get('depth_storage', {})
            if not storage.get('file') or not (path.parent/storage['file']).is_file():
                continue
            yield dict(feature_key=path.parent.name, shape=[storage['height'], storage['width'], 1],
                sample_count=storage['frames'], sample_rate_hz=storage.get('fps'),
                coordinate_frame='rectified_left_camera', unit=storage.get('unit'),
                invalid_value=storage.get('invalid_value'), format='hevc12' if storage.get('pixel_format') == 'gray12le' else storage.get('codec'))


def build(session, level='L1'):
    info = load(session/'meta/info.json')
    features = info['features']
    files = [p for p in session.rglob('*') if p.is_file()
             and p != session/'meta/readme.json'
             and not (p.parent == session/'meta' and p.name.startswith('.readme.json.'))]
    annotations = [load(p) for p in sorted((session/'annotation').glob('episode_*_structured.json'))]
    annotation = annotations[0] if annotations else {}
    source = load(session/'meta/guanglun_source.json', {})
    task_info = source.get('collection_metadata', {}).get('task_info', {})
    tasks = lines(session/'meta/tasks.jsonl')
    task_name = annotation.get('task_name') or task_info.get('task_name') or (tasks[0].get('task') if len(tasks) == 1 else None)
    if task_name:
        task_name = re.sub(r'[\W_]+', '_', task_name).strip('_')
    classification = annotation.get('scene_classification') or {}
    category, area = classification.get('category'), classification.get('area')
    environment = str(task_info.get('environment_id', '')).lower()
    scene_type = infer_scene_type(category, area, environment)
    cameras = []
    primary = 'observation.images.head_left_rectified'
    for key, f in features.items():
        if f.get('dtype') == 'video':
            h, w = f['shape'][:2]
            video_paths = sorted(p for p in files if p.suffix == '.mp4' and p.parent.name == key)
            cameras.append(dict(id=key, width=w, height=h, fps=f.get('fps', info.get('fps')),
                                sample_count=video_sample_count(video_paths)))
    main = next((c for c in cameras if c['id'] == primary), None)
    indexes = [p for p in files if p.name == 'master_index.parquet']
    frames = sum(pq.ParquetFile(p).metadata.num_rows for p in indexes) if indexes else info.get('total_frames')
    fps = info.get('fps')
    modalities = {}
    def add(group, stream):
        modalities.setdefault(group, {'streams': []})['streams'].append(stream)
    groups = {
        'hand_estimation': lambda k: k.startswith('observation.hand_') and ('points' in k or 'quaternions' in k),
        'head_trajectory': lambda k: k == 'observation.head_pose',
        'body_trajectory': lambda k: k in ('observation.body_pose', 'observation.upper_body_pose', 'observation.lower_body_pose'),
        'camera_trajectory': lambda k: k.endswith('_camera_pose'),
        'quality_flags': lambda k: k.startswith('observation.bad_frame_'),
        'eye_gaze': lambda k: 'gaze' in k,
        'imu': lambda k: 'imu' in k,
        'emg': lambda k: k.endswith('_emg'),
        'eeg': lambda k: k == 'observation.eeg',
    }
    for key in sorted({p.stem for p in files if p.suffix == '.parquet' and 'depth' not in p.relative_to(session).parts}):
        group = next((g for g, match in groups.items() if match(key)), None)
        if group is None:
            continue
        paths = [p for p in files if p.name == key+'.parquet']
        f = features.get(key, {})
        shape = f.get('shape')
        if shape is None:
            cols = [n for n in pq.read_schema(paths[0]).names if n not in ('timestamp', 'episode_index', 'frame_index', 'index', 'task_index')]
            shape = [len(cols)]
        stream = dict(feature_key=key, shape=shape,
                      sample_count=sum(pq.ParquetFile(p).metadata.num_rows for p in paths),
                      sample_rate_hz=f.get('fps', f.get('sample_rate')))
        for field in ('coordinate_frame', 'quaternion_order'):
            if f.get(field):
                stream[field] = f[field]
        add(group, stream)
    for stream in depth_streams(session):
        add('depth', stream)
    actions = set()
    objects = {}
    for a in annotations:
        actions.update(a.get('actions', {}).get('vocabulary', []))
        for o in a.get('objects', []):
            attrs = o.get('attributes') or {}
            objects[o['name']] = {'name': o['name'], 'attributes': {k: attrs.get(k) if isinstance(attrs, dict) else None for k in ('color', 'material', 'shape')}}
    eeg = emg = 0
    for group in ('eeg', 'emg'):
        count = 0
        for stream in modalities.get(group, {}).get('streams', []):
            if not stream['sample_count']:
                continue
            f = features.get(stream['feature_key'], {})
            names = f.get('names') or []
            count += (len([n for n in names if n.upper() not in {'VEOG','HEOG','EKG','ECG','EMG','TRIGGER'}])
                      if group == 'eeg' and names else stream['shape'][0])
        if group == 'eeg': eeg = count
        else: emg = count
    parts = [level, area or category or 'unknown', task_name or 'unknown', 'left',
             f"{main['width']}x{main['height']}" if main else 'unknown',
             f"{main['fps']:g}FPS" if main and main.get('fps') else 'unknown',
             f'{len(cameras)}views', f'EEG{eeg}', f'EMG{emg}']
    dataset_name = re.sub(r'[\W_]+', '_', '_'.join(parts)).strip('_')
    return lowercase_values(dict(dataset_name=dataset_name, hand_capture_type='bare_hand',
        scene=dict(type=scene_type, category=category, name=area), task_name=task_name,
        common_task=annotation.get('common_task'),
        task_description=annotation.get('task_description') or task_info.get('task_description'),
        actions=sorted(actions), objects=list(objects.values()),
        duration_s=frames/fps if frames is not None and fps else None,
        frame_count=frames, master_fps=fps, size_bytes=sum(p.stat().st_size for p in files), file_count=len(files),
        cameras=dict(count=len(cameras), primary_camera=primary if main else None,
                     synchronization_master=info.get('master_feature'), items=cameras),
        modalities=normalize_modalities(modalities)))


def discover(root):
    if (root/'meta/info.json').is_file():
        yield root
        return
    batches = [root] if root.name.startswith('batch_') else sorted(root.glob('batch_*'))
    for batch in batches:
        print('SCAN', batch, flush=True)
        for session in sorted(batch.iterdir()):
            if (session/'meta/info.json').is_file():
                yield session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--level', choices=['L1', 'L2', 'L3'], default='L1')
    args = parser.parse_args(argv)
    if not args.root.is_dir(): parser.error('Root directory does not exist')
    done = failed = 0
    for session in discover(args.root):
        tmp = None
        try:
            result = build(session, args.level)
            fd, tmp = tempfile.mkstemp(prefix='.readme.json.', dir=session/'meta')
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            os.replace(tmp, session/'meta/readme.json')
            print('WROTE', session/'meta/readme.json', flush=True)
            done += 1
        except Exception as exc:
            print('FAILED', session, str(exc), flush=True)
            failed += 1
        finally:
            if tmp: Path(tmp).unlink(missing_ok=True)
    print(json.dumps(dict(written=done, failed=failed)))
    return int(failed > 0 or done == 0)


if __name__ == '__main__':
    raise SystemExit(main())
