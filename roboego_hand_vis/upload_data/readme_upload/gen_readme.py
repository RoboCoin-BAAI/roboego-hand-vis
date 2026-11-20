#!/usr/bin/env python3
"""Generate a HumanCOIN dataset card from local metadata; never uploads."""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile

if __package__:
    from ._common import DEFAULT_IGNORE, plan_files, repo_id_for
else:
    from _common import DEFAULT_IGNORE, plan_files, repo_id_for

PROJECT_CONFIG = Path(__file__).resolve().parent / 'config' / 'project.json'
INDEX_FEATURES = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
PORTAL_METADATA = 'portal.yaml'
COLLECTION_DOCUMENTS = {'README.md', PORTAL_METADATA}


def read_json(path):
    with path.open(encoding='utf-8-sig') as f:
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
    if not (folder / 'meta' / 'info.json').is_file():
        return render_l1_readme(folder, project, namespace, allow, ignore)
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
             table(['Object'], [(o.get('name'),) for o in description.get('objects', [])]), '',
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
    lines += project_sections(project)
    return '\n'.join(lines)


def project_sections(project):
    lines = ['## Authors and contact', '', ', '.join(project.get('authors', [])), '',
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
    return lines


def l1_task_metadata(folder):
    """Read task cards, retaining deterministic first-task semantics."""
    if not folder.name.startswith('l1_'):
        raise ValueError('A collection must be an L1 dataset_name directory')
    tasks = []
    for path in sorted(folder.iterdir()):
        if path.name in ('.git', '.cache', '__pycache__', 'backup'):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink task path is not supported: {path}')
        if not path.is_dir():
            continue
        description = read_json(path / 'meta' / 'readme.json')
        if not (path / 'meta' / 'info.json').is_file():
            raise ValueError(f'Missing task info: {path}')
        if description.get('dataset_name') != folder.name:
            raise ValueError(f'Task dataset_name differs from its parent: {path}')
        if description.get('unique_task_name') != path.name:
            raise ValueError(f'Task unique_task_name differs from its directory: {path}')
        tasks.append((path, description))
    if not tasks:
        raise ValueError('L1 dataset contains no task directories')
    return tasks


def _total(tasks, key, integer=False):
    values = []
    for path, description in tasks:
        value = description.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or integer and not isinstance(value, int)):
            raise ValueError(f'Missing/invalid {key} in {path}/meta/readme.json')
        values.append(value)
    return sum(values)


def render_l1_readme(folder, project, namespace='HumanCOIN', allow=None, ignore=None):
    return render_l1_documents(folder, project, namespace, allow, ignore)[0]


def render_l1_documents(folder, project, namespace='HumanCOIN', allow=None, ignore=None):
    """One upload card per dataset_name; samples remain independent task roots."""
    if project.get('license', 'cc-by-nc-4.0') != 'cc-by-nc-4.0':
        raise ValueError('This release is configured for CC BY-NC 4.0')
    tasks = l1_task_metadata(folder)
    first, description = tasks[0]
    info = read_json(first / 'meta' / 'info.json')
    features = info.get('features')
    if not isinstance(features, dict) or not features:
        raise ValueError('First task info must contain feature definitions')
    name = description['dataset_name']
    repo_id = repo_id_for(folder, namespace)
    frames = _total(tasks, 'frame_count', integer=True)
    duration = _total(tasks, 'duration_s')
    # Each task card excludes itself, but is a real file in the uploaded collection.
    file_count = _total(tasks, 'file_count', integer=True) + len(tasks)
    size = _total(tasks, 'size_bytes', integer=True) + sum((p/'meta/readme.json').stat().st_size for p, _ in tasks)
    if allow or ignore:
        files = [p for p in plan_files(folder, allow=allow, ignore=ignore) if p not in COLLECTION_DOCUMENTS]
        file_count = len(files)
        size = sum((folder/p).stat().st_size for p in files)
    else:
        for p in folder.iterdir():
            if p.is_file() and p.name not in COLLECTION_DOCUMENTS and not any(fnmatch.fnmatchcase(p.name, pattern) for pattern in DEFAULT_IGNORE):
                file_count += 1
                size += p.stat().st_size
    actions = sorted({str(action) for _, d in tasks for action in d.get('actions', []) if action})
    objects = sorted({str(o.get('name')) for _, d in tasks for o in d.get('objects', []) if isinstance(o, dict) and o.get('name')})
    task_names = sorted({d['task_name'] for _, d in tasks if d.get('task_name')})
    common = description.get('common_task_name')
    primary = (description.get('cameras') or {}).get('primary_camera')
    shape = features.get(primary, {}).get('shape') or []
    named_resolution = re.search(r'_(\d+x\d+)(?:_|$)', name)
    resolution = (f'{shape[1]}x{shape[0]}' if len(shape) >= 2 else
                  named_resolution[1] if named_resolution else None)
    example_description = description.get('task_description') or description.get('task_name')
    keys = ' '.join(features).lower()
    modalities = [label for label, present in (
        ('RGB', any(ft.get('dtype') == 'video' and 'depth' not in key for key, ft in features.items())),
        ('Depth', 'depth' in keys), ('Hand Pose', any(key.startswith('observation.hand_') for key in features)),
        ('Body Pose', 'body_pose' in keys), ('Gaze', 'gaze' in keys),
        ('Eye Camera', 'eye_rgb' in keys), ('EMG', 'emg' in keys), ('EEG', 'eeg' in keys),
        ('Manus', description.get('hand_capture_type') == 'glove' or 'hand_skeleton' in keys),
        ('Audio', any(ft.get('dtype') == 'audio' for ft in features.values())), ('IMU', 'imu' in keys),
    ) if present]
    portal = {
        'schema_version': 1, 'repo_id': repo_id, 'dataset_name': name, 'level': 'L1',
        'license': 'cc-by-nc-4.0', 'scene': description.get('scene'), 'task_name': common,
        'task_labels': task_names, 'example_task_description': example_description,
        'example_recording': first.name, 'num_recordings': len(tasks), 'frame_count': frames,
        'duration_s': duration, 'file_count': file_count, 'size_bytes': size,
        'resolution': resolution, 'modalities': modalities, 'primary_camera': primary,
    }
    if primary and 'depth' not in primary.lower() and info.get('video_path'):
        video = first / info['video_path'].format(episode_chunk=0, episode_index=0, video_key=primary)
        if video.is_file() and video.resolve().is_relative_to(first.resolve()):
            relative_video = video.relative_to(folder).as_posix()
            if not (allow or ignore) or relative_video in files:
                portal['preview_video'] = relative_video
    lines = ['---', 'license: cc-by-nc-4.0', 'gated: true',
             'extra_gated_prompt: "Please provide your name, email address and intended use to request access to HumanCOIN datasets."',
             'extra_gated_fields:', '  Name: text', '  Email: text', '  Intended use: text',
             'viewer: false', 'language: [en, zh]',
             'pretty_name: ' + json.dumps(name, ensure_ascii=False),
             'tags: ' + json.dumps(['HumanCOIN', 'L1', 'egocentric', 'multi-sensor'] + ([common] if common else [])),
             '---', '', '# ' + name, '', '## Dataset description', '',
             'Read with [HumanCOIN](https://github.com/FlagOpen/HumanCOIN).', '',
             '```bash', 'pip install -U "humancoin>=0.1.1"', '```', '',
             'This L1 dataset groups independent task recordings under one repository. Each `unique_task_name/` directory is a complete HumanCOIN recording with its own metadata and master timeline.', '',
             '**Example task description (first task):** ' + cell(example_description), '',
             table(['Property', 'Value'], [
                 ('Repository ID', repo_id), ('Data tier', 'L1'),
                 ('Task category', common), ('Task labels', ', '.join(task_names)),
                 ('Scene', scene_label(description.get('scene'))),
                 ('Hand capture', description.get('hand_capture_type')),
                 ('Tasks', len(tasks)), ('Master frames', frames),
                 ('Master frame rate (Hz)', info.get('fps')),
                 ('Duration', f'{duration:.2f} s ({duration/3600:.3f} h)'),
                 ('Data format', 'HumanCOIN-LeRobot'),
                 ('Files (excluding README.md and portal.yaml)', file_count),
                 ('Size (excluding README.md and portal.yaml)', f'{size/1024**2:.2f} MiB'),
             ]), '',
             'Task count is the number of `unique_task_name/` recordings, including recordings with the same task label. Frame counts, durations and file statistics are summed across all tasks. File statistics include each task\'s `meta/readme.json` and exclude `README.md` and `portal.yaml` at the repository root.', '',
             'Scene, tier, example description and shared stream/storage metadata below come from the first task in directory-name order.', '',
             '## Actions and objects', '',
             '**Atomic actions (deduplicated union):** ' + cell(', '.join(a.replace('_', ' ') for a in actions)), '',
             table(['Object'], [(o,) for o in objects]), '',
             '## Sensor streams', '',
             'Shared feature definitions from the first task. Video shapes use HWC order; loaded images use CHW. Independent sensor rates and query windows are preserved. Per-stream frame/sample counts are not aggregated here.', '']
    streams = []
    sensor_streams = []
    for key, ft in features.items():
        if key in INDEX_FEATURES or key.startswith('observation.bad_frame_'):
            continue
        supported = (ft.get('dtype') == 'video' or key.startswith('observation.hand_')
                     or key in ('observation.head_pose', 'observation.relative_head_pose', 'observation.body_pose', 'observation.upper_body_pose', 'observation.lower_body_pose')
                     or key.endswith('_camera_pose') or 'gaze' in key or 'imu' in key)
        if not supported:
            continue
        kind = 'depth video' if ft.get('info', {}).get('video.is_depth_map') is True else ft.get('dtype')
        rate = ft.get('fps', ft.get('sample_rate'))
        sensor_streams.append({
            'feature_key': key, 'dtype': ft.get('dtype'),
            'shape': ft.get('shape'), 'sample_rate_hz': rate,
        })
        if isinstance(rate, (int, float)):
            rate = f'{rate:g}'
        streams.append((key, kind, ft.get('shape'), rate, ft.get('window', 'nearest')))
    portal['sensor_streams'] = sensor_streams
    lines += [table(['Feature', 'Type', 'Shape', 'Rate (Hz)', 'Query window'], streams), '',
              '**Primary camera:** ' + cell(description.get('cameras', {}).get('primary_camera')), '',
              '## Storage and time alignment', '',
              'Each task has the same relative layout; timeline indices restart inside each task:', '',
              '```text', name + '/', '├── README.md', '└── <unique_task_name>/',
              '    ├── annotation/', '    ├── calibrations/', '    ├── data/', '    ├── meta/', '    └── videos/', '```', '',
              table(['Component', 'Path template'], [(key, '<unique_task_name>/' + info[key]) for key in ('index_path', 'data_path', 'video_path') if info.get(key)]), '',
              'Reading `ds[i]` queries each feature using that task\'s master-index timestamp. A query window selects samples in `(t + start, t + end]`; `nearest` selects the closest sample. Task labels are stored in each task\'s `meta/tasks.jsonl`.', '',
              '## Loading', '', 'Load a task directory inside the downloaded repository:', '',
              '```python', 'from pathlib import Path', 'from humancoin import HumanCOINDataset', '',
              f'repo_id = {json.dumps(repo_id)}', 'dataset_root = Path("data") / repo_id',
              'task_dirs = sorted(p for p in dataset_root.iterdir() if (p / "meta/info.json").is_file())', '',
              'for task_dir in task_dirs:', '    ds = HumanCOINDataset(repo_id=repo_id, root=task_dir, video_backend="pyav")',
              '    item = ds[0]', '    print(task_dir.name, len(ds), item.keys())', '```', '']
    if any(ft.get('info', {}).get('video.is_depth_map') is True for ft in features.values()):
        lines += ['Depth videos are read as native single-channel values without RGB conversion or normalization. Depth units, scale and encoding are declared in each task\'s `meta/info.json`. Supplementary masks and depth metadata, when provided, are stored in that task\'s annotation directory.', '']
    lines += ['## Annotations and calibration', '', 'Shared artifact paths, taken from the first task:', '']
    for directory in ('annotation', 'calibrations'):
        for p in sorted((first/directory).rglob('*')):
            if p.is_file():
                lines.append('- `' + '<unique_task_name>/' + p.relative_to(first).as_posix() + '`')
    lines += ['', *project_sections(project)]
    return '\n'.join(lines), portal


def render_portal_yaml(metadata):
    import yaml

    return yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)


def generate_portal_yaml(folder, metadata):
    """Extract the first frame of the selected preview video and serialize its path."""
    content = render_portal_yaml(metadata)
    if not metadata.get('preview_video'):
        return content
    ffmpeg = shutil.which('ffmpeg') or shutil.which('ffmpeg', path=str(Path(sys.executable).parent))
    if not ffmpeg:
        raise ValueError('FFmpeg is required to generate preview.jpg')
    target = folder / 'preview.jpg'
    with NamedTemporaryFile(dir=folder, prefix='.preview.', suffix='.jpg', delete=False) as file:
        temporary = Path(file.name)
    try:
        subprocess.run(
            [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
             '-i', str(folder / metadata['preview_video']), '-map', '0:v:0',
             '-frames:v', '1', '-threads', '1', '-q:v', '2', str(temporary)],
            check=True, capture_output=True, text=True,
        )
        if not temporary.stat().st_size:
            raise ValueError(f'FFmpeg produced no preview frame: {folder}')
        temporary.chmod(0o644)
        temporary.replace(target)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f'FFmpeg preview failed for {folder}: {exc.stderr.strip()}') from exc
    finally:
        temporary.unlink(missing_ok=True)
    return render_portal_yaml({**metadata, 'preview_image': target.name})


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
        is_collection = not (folder / 'meta' / 'info.json').is_file()
        portal_output = folder / PORTAL_METADATA if is_collection else None
        if portal_output and output == portal_output.resolve():
            raise ValueError('README output must differ from portal.yaml')
        for path in (output, portal_output) if is_collection else (output,):
            if path.exists() and not args.force:
                raise ValueError(f'Output exists; use --force to replace: {path}')
        project = read_json(args.project_config)
        if is_collection:
            content, metadata = render_l1_documents(folder, project, args.namespace, args.allow, args.ignore)
            portal_content = generate_portal_yaml(folder, metadata)
        else:
            content = render_readme(folder, project, args.namespace, args.allow, args.ignore)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding='utf-8')
        print(f'Generated: {output}')
        if portal_output:
            portal_output.write_text(portal_content, encoding='utf-8')
            print(f'Generated: {portal_output}')
        return 0
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        print(f'Generation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
