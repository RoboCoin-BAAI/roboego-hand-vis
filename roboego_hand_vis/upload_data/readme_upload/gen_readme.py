#!/usr/bin/env python3
"""Generate a HumanCOIN dataset card from local metadata; never uploads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__:
    from ._common import plan_files, repo_id_for
else:
    from _common import plan_files, repo_id_for

PROJECT_CONFIG = Path(__file__).resolve().parent / 'config' / 'project.json'
INDEX_FEATURES = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}


def read_json(path):
    with path.open(encoding='utf-8') as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def cell(value):
    if value is None:
        return 'Not specified'
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace('|', '&#124;').replace('\n', '<br>').replace('\r', '')


def scene_label(scene):
    if not isinstance(scene, dict):
        return scene
    return ' / '.join(str(scene[k]).replace('_', ' ').capitalize()
                      for k in ('type', 'category', 'name') if scene.get(k)) or None


def attributes_label(attributes):
    if not isinstance(attributes, dict):
        return attributes or '—'
    return ', '.join(f"{key.replace('_', ' ')}: {value}"
                     for key, value in attributes.items() if value is not None and value != '') or '—'


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows]
    return '\n'.join(lines)


def tasks_from(folder):
    path = folder / 'meta' / 'tasks.jsonl'
    if not path.exists():
        return []
    tasks = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('task') and row['task'] not in tasks:
                tasks.append(row['task'])
    return tasks


def render_readme(folder: Path, project: dict, namespace='HumanCOIN', allow=None, ignore=None) -> str:
    folder = folder.expanduser().resolve()
    description = read_json(folder / 'meta' / 'readme.json')
    info = read_json(folder / 'meta' / 'info.json')
    name = description.get('dataset_name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('meta/readme.json must contain a nonempty dataset_name')
    repo_id = repo_id_for(folder, namespace)
    features = info.get('features')
    if not isinstance(features, dict) or not features:
        raise ValueError('meta/info.json must contain feature definitions')
    files = [p for p in plan_files(folder, allow=allow, ignore=ignore) if p != 'README.md']
    size = sum((folder / p).stat().st_size for p in files)
    tasks = tasks_from(folder)
    license_id = project.get('license', 'cc-by-nc-4.0')
    if license_id != 'cc-by-nc-4.0':
        raise ValueError('This release is configured for CC BY-NC 4.0')
    tier = name.split('_', 1)[0].upper()
    enable_viewer = tier in ('L2', 'L3')
    cameras = description.get('cameras', {})
    task_name = description.get('task_name')
    pretty_name = task_name.replace('_', ' ').capitalize() if isinstance(task_name, str) and task_name else name
    tags = ['HumanCOIN']
    if tier in ('L1', 'L2', 'L3'):
        tags.append(tier)
    tags.extend(['egocentric', 'multi-sensor'])
    if isinstance(task_name, str) and task_name:
        tags.append(task_name)
    lines = ['---', 'license: cc-by-nc-4.0', 'gated: true',
             'extra_gated_prompt: "Please provide your name, email address and intended use to request access to HumanCOIN datasets."',
             'extra_gated_fields:',
             '  Name: text',
             '  Email: text',
             '  Intended use: text',
             f'viewer: {str(enable_viewer).lower()}', 'language: [en, zh]',
             'pretty_name: ' + json.dumps(pretty_name, ensure_ascii=False),
             'tags: ' + json.dumps(tags, ensure_ascii=False),
             '---', '', '# ' + name, '',
             '## Dataset description',
             '',
             'Read with [HumanCOIN](https://github.com/FlagOpen/HumanCOIN).',
             '',
             '```bash',
             'pip install -U "humancoin>=0.1.1"',
             '```',
             '',
             'A HumanCOIN multi-sensor recording of human activity. Sensor streams retain independent sampling rates and timestamps and are queried against a shared master timeline.', '',
             '**' + str(description.get('task_description') or description.get('task_name') or 'Task description not specified.') + '**', '',
             table(['Property', 'Value'], [
                 ('Repository ID', repo_id),
                 ('Data tier', tier if tier in ('L1', 'L2', 'L3') else None),
                 ('Task labels', ', '.join(tasks) if tasks else description.get('task_name')),
                 ('Task name', description.get('task_name')),
                 ('Scene', scene_label(description.get('scene'))),
                 ('Hand capture', description.get('hand_capture_type')),
                 ('Episodes', info.get('total_episodes')),
                 ('Master frames', info.get('total_frames')),
                 ('Master frame rate (Hz)', info.get('fps')),
                 ('Duration', f"{float(description['duration_s']):.2f} s" if description.get('duration_s') is not None else None),
                 ('Data format', 'HumanCOIN-LeRobot'),
                 ('Files (excluding README.md)', len(files)),
                 ('Size (excluding README.md)', f'{size / 1024**2:.2f} MiB'),
             ]), '',
             '## Actions and objects', '',
             '**Atomic actions:** ' + cell(', '.join(str(action).replace('_', ' ') for action in description.get('actions', []))), '',
             table(['Object', 'Attributes'], [(o.get('name'), attributes_label(o.get('attributes'))) for o in description.get('objects', [])]), '',
             '## Sensor streams', '',
             'Shapes describe one stored sample in metadata. Video shapes use HWC order; loaded images use CHW. Sensor rates differ from the master frame rate.', '']
    sample_counts = {}
    for camera in description.get('cameras', {}).get('items', []):
        sample_counts[camera.get('id')] = camera.get('sample_count')
    for modality in description.get('modalities', {}).values():
        for stream in modality.get('streams', []):
            sample_counts[stream.get('feature_key')] = stream.get('sample_count')
    rows = []
    for key, feature in features.items():
        if key in INDEX_FEATURES:
            continue
        kind = feature.get('dtype')
        if feature.get('info', {}).get('video.is_depth_map') is True:
            kind = 'depth video'
        rate = feature.get('sample_rate') if kind == 'audio' else feature.get('fps')
        rows.append((key, kind, feature.get('shape'), rate,
                     sample_counts.get(key), feature.get('window', 'nearest')))
    lines += [table(['Feature', 'Type', 'Shape', 'Rate (Hz)', 'Samples (metadata)', 'Query window'], rows), '']
    cameras = description.get('cameras', {})
    lines += ['**Primary camera:** ' + cell(cameras.get('primary_camera')), '',
              '**Master timeline:** `master_index.parquet`', '']
    depth = [k for k, ft in features.items() if ft.get('info', {}).get('video.is_depth_map') is True]
    lines += ['## Storage and time alignment', '',
              table(['Component', 'Path template'], [(k, info.get(k)) for k in ('index_path', 'data_path', 'video_path', 'audio_path') if info.get(k)]), '',
              'Reading ds[i] uses the master-index timestamp to query all sensor features. A window [start, end] selects samples in (t + start, t + end]; nearest selects one sample. Windows can return different numbers of samples, including empty arrays near recording boundaries or gaps. Task labels are read from meta/tasks.jsonl.', '',
              '**Declared episode splits:** ' + cell(info.get('splits')), '',
              '## Loading', '', 'After installation, load the downloaded dataset from its local directory:', '',
              '```python', 'from humancoin import HumanCOINDataset', '',
              'ds = HumanCOINDataset(', f'    repo_id={json.dumps(repo_id)},',
              f'    root={json.dumps("data/" + repo_id)},  # local downloaded dataset directory',
              ')', 'item = ds[0]', 'print(item.keys())', '```', '']
    if depth:
        lines += ['HumanCOIN reads depth as native single-channel values without RGB conversion or normalization. Depth is stored at 12-bit precision in millimeters and decoded as `uint16`; encoding is lossy, so this does not imply 1 mm measurement accuracy. Supplementary depth metadata and valid-pixel masks are stored under `annotation/chunk-*/<depth_feature>/episode_*/`.', '']
    annotations = [p for p in files if p in ('annotation/episode_000000_structured.json', 'annotation/quality_report.json')]
    lines += ['## Annotations and quality information', '']
    if annotations:
        lines += ['Selected annotation artifacts:', ''] + ['- ' + cell(p) for p in annotations] + ['']
    reports = [p for p in files if p.startswith('calibrations/')]
    lines += ['See the included quality and calibration artifacts for checks and assumptions. Their presence does not imply that every check passed.', '']
    lines += ['- ' + cell(p) for p in reports] + ['']
    lines += ['## Authors and contact', '', ', '.join(project.get('authors', [])), '',
              '**Institution:** ' + cell(project.get('institution')), '',
              '**Contact:** ' + cell(project.get('contact')), '',
              '**Project:** ' + str(project.get('homepage', '')), '',
              '## License', '',
              'This dataset and its documentation are released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). This dataset card uses the license selected for this release.', '',
              '## Citation', '', '```bibtex']
    citation = project.get('citation', {})
    lines += ['@techreport{' + citation.get('key', 'humancoin') + ',',
              '  title = {' + citation.get('title', 'HumanCOIN') + '},',
              '  author = {' + ' and '.join(project.get('authors', [])) + '},',
              '  institution = {' + str(project.get('institution', '')) + '},',
              '  year = {' + str(citation.get('year', '')) + '},',
              '  url = {' + str(project.get('homepage', '')) + '}', '}', '```', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, help='Default: <dataset>/README.md')
    p.add_argument('--project-config', type=Path, default=PROJECT_CONFIG)
    p.add_argument('--namespace', default='HumanCOIN')
    p.add_argument('--allow', nargs='+')
    p.add_argument('--ignore', nargs='+')
    p.add_argument('--force', action='store_true', help='Replace an existing output file')
    args = p.parse_args(argv)
    try:
        folder = args.dataset.expanduser().resolve()
        output = (args.output.expanduser() if args.output else folder / 'README.md').resolve()
        if output.exists() and not args.force:
            raise ValueError(f'Output exists; use --force to replace: {output}')
        project = read_json(args.project_config)
        content = render_readme(folder, project, args.namespace, args.allow, args.ignore)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding='utf-8')
        print(f'Generated: {output}')
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f'Generation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
