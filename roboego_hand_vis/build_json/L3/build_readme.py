"""Build compact L3 meta/readme.json files without model calls.

Usage: python build_readme.py --root SESSION_OR_BATCH_OR_DATASET
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_metadata import build as build_manifest, read_json
from video_samples import video_sample_count
from readme_schema import lowercase_values, normalize_modalities
from base.build_readme import depth_streams, infer_scene_type


def build(session, context=None):
    context = dict(context or {})
    context['level'] = 'L3'
    manifest = build_manifest(session, context, read_collection_metadata=False)
    cameras = manifest['cameras']
    annotations = [read_json(p) for p in sorted((session/'annotation').glob('episode_*_structured.json'))]
    classification = next((a['scene_classification'] for a in annotations if a.get('scene_classification')), {})
    category = classification.get('category')
    scene_type = infer_scene_type(category, classification.get('area'))
    scene = dict(type=scene_type, category=category, name=classification.get('area'))
    modalities = {}
    for group, data in manifest['modalities'].items():
        streams = []
        for source in data['streams']:
            if not source['data_paths']:
                continue
            stream = dict(feature_key=source['feature_key'], shape=source['definition'].get('shape'),
                          sample_count=source['sample_count'], sample_rate_hz=source['sample_rate_hz'])
            for field in ('coordinate_frame', 'quaternion_order'):
                value = source['definition'].get(field)
                if value:
                    stream[field] = value
            streams.append(stream)
        if streams:
            modalities[group] = {'streams': streams}
    stored_depth = list(depth_streams(session))
    if stored_depth:
        modalities.setdefault('depth', {'streams': []})['streams'].extend(stored_depth)
    objects = {}
    for obj in [obj for annotation in annotations for obj in annotation.get('objects', [])]:
        attrs = obj.get('attributes') or {}
        objects[obj['name']] = dict(name=obj['name'], attributes={
            k: attrs.get(k) if isinstance(attrs, dict) else None for k in ('color', 'material', 'shape')})
    vocab = {v for a in annotations for v in a.get('actions', {}).get('vocabulary', [])}
    task_name = next((a.get('task_name') for a in annotations if a.get('task_name')), None)
    if task_name:
        task_name = re.sub(r'[\W_]+', '_', task_name).strip('_')
    items = [{k: c[k] for k in ('id', 'width', 'height', 'fps')} for c in cameras['items']]
    for item, camera in zip(items, cameras['items']):
        item['sample_count'] = video_sample_count([session/p for p in camera['video_paths']])
    main = next((c for c in items if c['id'] == cameras['primary_camera']), None)
    eeg, emg = (manifest['channel_counts'][k] for k in ('eeg', 'emg'))
    parts = ['L3', scene['name'] or scene['category'] or 'unknown', task_name or 'unknown',
             cameras['primary_camera_label'], f"{main['width']}x{main['height']}" if main else 'unknown',
             f"{main['fps']:g}FPS" if main and main.get('fps') else 'unknown',
             f"{len(items)}views", f'EEG{eeg}', f'EMG{emg}']
    result = dict(dataset_name=re.sub(r'[\W_]+', '_', '_'.join(parts)).strip('_'),
                  hand_capture_type='glove' if 'glove' in modalities else 'unknown',
                  scene=scene, task_name=task_name,
                  common_task=next((a.get('common_task') for a in annotations if a.get('common_task')), None),
                  task_description=next((a.get('task_description') for a in annotations if a.get('task_description')), None),
                  actions=sorted(vocab),
                  objects=list(objects.values()))
    result.update({k: manifest['statistics'][k] for k in
                   ('duration_s', 'frame_count', 'master_fps', 'size_bytes', 'file_count')})
    result.update(cameras=dict(count=len(items), primary_camera=cameras['primary_camera'] if main else None,
                              synchronization_master=cameras['synchronization_master'], items=items),
                  modalities=normalize_modalities(modalities))
    return lowercase_values(result)


def discover(root):
    if (root/'meta/info.json').is_file():
        yield root
    else:
        for path in sorted(root.rglob('meta/info.json')):
            yield path.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--context', type=Path, help='Optional manual L3 semantic fields')
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error('Root directory does not exist')
    context = read_json(args.context) if args.context else {}
    done = failed = 0
    for session in discover(args.root):
        tmp = None
        try:
            result = build(session, context)
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
            if tmp:
                Path(tmp).unlink(missing_ok=True)
    print(json.dumps(dict(written=done, failed=failed)))
    return int(failed > 0 or done == 0)


if __name__ == '__main__':
    raise SystemExit(main())
