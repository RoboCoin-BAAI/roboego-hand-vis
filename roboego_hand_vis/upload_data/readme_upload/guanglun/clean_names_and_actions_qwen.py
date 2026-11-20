#!/usr/bin/env python3
"""Prepare one object/action replacement bundle, then apply it after scene cleaning."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import stat
import sys

try:
    from . import clean_object_names_qwen as objects
    from . import clean_action_vocabulary_qwen as actions
    from .reclassify_scenes_qwen import discover, review_modules, save_json
    from .apply_scene_classification import atomic_write
except ImportError:
    import clean_object_names_qwen as objects
    import clean_action_vocabulary_qwen as actions
    from reclassify_scenes_qwen import discover, review_modules, save_json
    from apply_scene_classification import atomic_write

BASE = Path(__file__).resolve().parent


def source_fields(data):
    entries = data.get('objects')
    if not isinstance(entries, list) or not all(isinstance(row, dict) and isinstance(row.get('name'), str) for row in entries):
        raise ValueError('Expected existing objects[].name strings')
    return dict(object_names=[row['name'] for row in entries],
                action_vocabulary=actions.get_names(data), task_description=data.get('task_description'))


def read_only_process(function, path, args, endpoint, vocabulary):
    if args.apply:
        raise ValueError('Prepare must never apply annotations')
    try:
        return function(path, args, endpoint, vocabulary)
    except ValueError as exc:
        # A parallel scene writer can change category/area during the API call.
        # Both cleaners save the completion before their source check, so this
        # second read reuses that response rather than reissuing a request.
        if str(exc) != 'Annotation changed concurrently':
            raise
        return function(path, args, endpoint, vocabulary)


def prepare_one(path, args, endpoint, object_words, action_words):
    before = source_fields(json.loads(path.read_text(encoding='utf-8-sig')))
    object_args = copy.copy(args)
    object_args.apply = False
    object_args.report_dir = args.state_dir / 'objects'
    action_args = copy.copy(args)
    action_args.apply = False
    action_args.report_dir = args.state_dir / 'actions'
    obj = read_only_process(objects.process, path, object_args, endpoint, object_words)
    act = read_only_process(actions.process, path, action_args, endpoint, action_words)
    latest = source_fields(json.loads(path.read_text(encoding='utf-8-sig')))
    if latest != before:
        raise ValueError('Object/action labels or task evidence changed during prepare')
    after_objects = list(before['object_names'])
    for change in obj.get('changes', []):
        if after_objects[change['index']] != change['before']:
            raise ValueError('Object mapping does not match the source index')
        after_objects[change['index']] = change['after']
    return dict(path=path.relative_to(args.root).as_posix(), status='prepared' if args.dry_run else 'done',
                before=before, after=dict(object_names=after_objects, action_vocabulary=act['after']),
                object_changes=obj.get('changes', []), action_mapping=act.get('mapping', []),
                unresolved_objects=obj.get('review', []), unresolved_actions=act.get('review', []),
                new_object_names=obj.get('new_names', []), new_action_verbs=act.get('new_verbs', []))


def prepare(args):
    args.root, args.output, args.state_dir = args.root.resolve(), args.output.resolve(), args.state_dir.resolve()
    for path in (args.output, args.state_dir):
        if path == args.root or args.root in path.parents:
            raise ValueError('Output and state directory must be outside the dataset')
    if args.output == args.state_dir or args.output in args.state_dir.parents:
        raise ValueError('Output file conflicts with state directory')
    if args.argus_root is None:
        args.argus_root = next((parent / 'argus' for parent in BASE.parents if (parent / 'argus/review/qwen_endpoint.py').is_file()), None)
    if args.argus_root is None:
        raise ValueError('Specify --argus-root')
    endpoint, overwrite = review_modules(args.argus_root.resolve())
    if not args.dry_run:
        endpoint.settings(args.model)
    object_words = {entry['name'] for entry in json.loads(args.object_vocabulary.read_text(encoding='utf-8-sig'))['entries']}
    table = json.loads(args.action_vocabulary.read_text(encoding='utf-8-sig'))
    action_words = {key: [entry['verb'] for entry in entries] for key, entries in table['groups'].items()}
    args.only_other = False
    paths = discover(args)
    with overwrite.job_lock(args.state_dir):
        args.state_dir.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        header = dict(version=1, root=str(args.root), model=args.model, status='running',
                      scope=['objects[].name', 'actions.vocabulary'])
        save_json(args.output, dict(**header, records=[]))
        records = []
        with (args.state_dir / 'prepare_progress.jsonl').open('w', encoding='utf-8') as progress:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(prepare_one, path, args, endpoint, object_words, action_words): path for path in paths}
                for future in as_completed(futures):
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = dict(path=futures[future].relative_to(args.root).as_posix(), status='error', error=f'{type(exc).__name__}: {exc}')
                    records.append(record)
                    progress.write(json.dumps(record, ensure_ascii=False) + '\n')
                    progress.flush()
                    event = dict(completed=len(records), total=len(paths), path=record['path'], status=record['status'])
                    if 'error' in record:
                        event['error'] = record['error']
                    print(json.dumps(event), flush=True)
        summary = dict(selected=len(paths), errors=sum(row['status'] == 'error' for row in records),
                       objects_changed=sum(len(row.get('object_changes', [])) for row in records),
                       action_files_changed=sum(row['before']['action_vocabulary'] != row['after']['action_vocabulary']
                           for row in records if row['status'] == 'done'))
        header['status'] = 'preview' if args.dry_run else 'complete'
        save_json(args.output, dict(**header, summary=summary, records=sorted(records, key=lambda row: row['path'])))
        print(json.dumps(dict(output=str(args.output), **summary)), flush=True)
    return 1 if summary['errors'] else 0


def apply_record(root, record, report_dir):
    relative = Path(record['path'])
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.json':
        raise ValueError('Unsafe bundle path')
    path = root / relative
    if path.is_symlink() or root not in path.resolve().parents:
        raise ValueError('Unsafe annotation target')
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    current = source_fields(data)
    before, after = record['before'], record['after']
    if current['task_description'] != before['task_description']:
        raise ValueError('Task description changed since model classification')
    for key in ('object_names', 'action_vocabulary'):
        if not isinstance(after[key], list) or not all(isinstance(value, str) and value for value in after[key]):
            raise ValueError('Invalid target names in replacement bundle')
        if current[key] != before[key] and current[key] != after[key]:
            raise ValueError('Source labels changed; refusing stale replacements: ' + key)
    if len(after['object_names']) != len(before['object_names']):
        raise ValueError('Bundle changed object count')
    revised = raw
    if current['object_names'] != after['object_names']:
        revised = objects.patch_names(revised, [dict(index=i, name=name) for i, name in enumerate(after['object_names'])])
    if current['action_vocabulary'] != after['action_vocabulary']:
        revised = actions.patch_vocabulary(revised, after['action_vocabulary'])
    expected = copy.deepcopy(data)
    for entry, name in zip(expected['objects'], after['object_names']):
        entry['name'] = name
    expected['actions']['vocabulary'] = after['action_vocabulary']
    if json.loads(revised.decode('utf-8-sig')) != expected:
        raise ValueError('Unexpected change outside the two requested fields')
    changed = revised != raw
    if changed:
        backup = report_dir / 'original_annotations' / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            if backup.read_bytes() != raw:
                raise ValueError('Existing backup differs from current annotation; inspect before resuming')
        else:
            with backup.open('xb') as stream:
                stream.write(raw)
        if path.read_bytes() != raw:
            raise ValueError('Annotation changed during apply; run after scene cleaning finishes')
        atomic_write(path, revised, stat.S_IMODE(path.stat().st_mode))
        if path.read_bytes() != revised:
            raise ValueError('Write-back verification failed')
    return dict(path=record['path'], status='done', changed=changed)


def apply_bundle(args):
    bundle = json.loads(args.bundle.read_text(encoding='utf-8-sig'))
    if bundle.get('version') != 1 or bundle.get('status') != 'complete' or bundle.get('scope') != ['objects[].name', 'actions.vocabulary']:
        raise ValueError('Expected a completed replacement bundle, not a preview/running result')
    root = (args.root or Path(bundle['root'])).resolve()
    report_dir = args.report_dir.resolve()
    if not root.is_dir() or report_dir == root or root in report_dir.parents:
        raise ValueError('Data root must exist; apply report directory must be outside dataset')
    records = bundle['records']
    if len(records) != len({record['path'] for record in records}):
        raise ValueError('Duplicate annotation paths in bundle')
    report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for record in records:
        if record['status'] != 'done':
            results.append(dict(path=record['path'], status='error', error='Preparation failed; no replacement attempted'))
            continue
        try:
            results.append(apply_record(root, record, report_dir))
        except Exception as exc:
            results.append(dict(path=record['path'], status='error', error=f'{type(exc).__name__}: {exc}'))
    summary = dict(selected=len(records), changed=sum(bool(row.get('changed')) for row in results),
                   errors=sum(row['status'] == 'error' for row in results))
    save_json(report_dir / 'apply_report.json', dict(summary=summary, records=results))
    print(json.dumps(summary), flush=True)
    return 1 if summary['errors'] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if sys.version_info < (3, 10):
        parser.error('Python 3.10+ required')
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('prepare', help='Read-only model cleaning, save one replacement JSON bundle')
    generate.add_argument('--root', type=Path, required=True)
    generate.add_argument('--output', type=Path, required=True)
    generate.add_argument('--state-dir', type=Path)
    generate.add_argument('--argus-root', type=Path)
    generate.add_argument('--start', type=int)
    generate.add_argument('--end', type=int)
    generate.add_argument('--batches', nargs='+')
    generate.add_argument('--limit', type=int, default=0)
    generate.add_argument('--workers', type=int, default=1)
    generate.add_argument('--model', default='qwen3.7-plus')
    generate.add_argument('--timeout', type=float, default=180)
    generate.add_argument('--max-tokens', type=int, default=4096)
    generate.add_argument('--retry-failed', action='store_true')
    generate.add_argument('--dry-run', action='store_true')
    generate.add_argument('--object-vocabulary', type=Path, default=BASE / 'object_vocabulary.json')
    generate.add_argument('--action-vocabulary', type=Path, default=BASE / 'action_vocabulary.json')
    apply = commands.add_parser('apply', help='After scene cleaning, replace only object names/action vocabulary; no API')
    apply.add_argument('--bundle', type=Path, required=True)
    apply.add_argument('--root', type=Path, help='Optional dataset relocation; source labels still verified')
    apply.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'apply':
        return apply_bundle(args)
    if args.workers < 1 or args.limit < 0 or args.timeout <= 0 or args.max_tokens < 64:
        parser.error('Invalid worker/limit/timeout/token setting')
    if args.batches and (args.start is not None or args.end is not None):
        parser.error('Choose --batches or --start/--end')
    if os.environ.get('ARGUS_ESTIMATED_CNY_LEDGER') and not args.dry_run:
        parser.error('This script does not implement ARGUS_ESTIMATED_CNY_LEDGER accounting')
    args.state_dir = args.state_dir or args.output.parent / (args.output.stem + '_state')
    args.apply = False
    return prepare(args)


if __name__ == '__main__':
    raise SystemExit(main())
