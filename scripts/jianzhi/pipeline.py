"""Download -> rectify -> structured annotation -> verified upload -> local cleanup."""
import argparse
import csv
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roboego_hand_vis.download_data.base.run import read_batch
from roboego_hand_vis.upload_data.ks3 import KS3, atomic_json, prefix

DESTINATION = 'ks3://baai-eai-pwm/jianzhi/wait_for_publish/'
BASE = Path('/share/project/liuyou/humancoin')


def annotation_command(args, work):
    return [args.python, '-m', 'review.jianzhi.structure_existing', '--source', str(work),
            '--workers', '1', '--model', args.model]


def build_readme(args, work, logs):
    relocate_runtime_files(work, logs / 'runtime_artifacts')
    run_stage([args.python, '-m', 'roboego_hand_vis.build_json.base.build_readme',
               '--root', str(work)], logs / 'build_readme.log', Path(__file__).resolve().parents[2])
    result = json.loads((work / 'meta/readme.json').read_text(encoding='utf-8'))
    if not isinstance(result, dict) or not result.get('dataset_name'):
        raise ValueError('Missing or invalid generated meta/readme.json')


def file_sizes(root):
    if root.is_symlink():
        raise ValueError('Dataset must not be a symlink')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Refusing symlink: {path}')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.stat().st_size
    return files


def verify_remote(client, destination, files):
    actual = {x['uri'][len(destination):]: x['size'] for x in client.objects(destination)
              if x['uri'].startswith(destination) and not x['uri'].endswith('/')}
    if not files or any(actual.get(name) != size for name, size in files.items()):
        raise ValueError('Remote upload verification failed; local data retained')


def relocate_runtime_files(work, external):
    """Keep recognized execution artifacts out of the published dataset."""
    work, external = Path(work).resolve(), Path(external).resolve()
    if external.is_relative_to(work) or work.is_relative_to(external):
        raise ValueError('Runtime archive must be separate from dataset')
    candidates = list(work.glob('.rectification.partial-*'))
    candidates += list((work/'annotation').glob('.*.execution.lock'))
    candidates += list((work/'annotation').glob('.*_text.claim.json'))
    candidates += list((work/'videos').glob('chunk-*/*/*.ffmpeg.log'))
    for path in candidates:
        if path.is_symlink():
            raise ValueError('Unexpected runtime symlink: '+str(path))
        target = external / path.relative_to(work)
        if target.exists():
            from uuid import uuid4
            target = target.with_name(target.name+'.'+uuid4().hex)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))


def cleanup(work, local_root, expected_files):
    base = Path(local_root).resolve()
    work = Path(work)
    resolved = work.resolve()
    if (work.is_symlink() or resolved == base or not resolved.is_relative_to(base)
            or len(resolved.relative_to(base).parts) != 2
            or not re.fullmatch(r'batch_\d+', resolved.parent.name)
            or not re.fullmatch(r'[0-9a-f]{32}', resolved.name)):
        raise ValueError(f'Unsafe cleanup path: {work}')
    if work.exists():
        if file_sizes(work) != expected_files:
            raise ValueError('Local files changed after upload; retaining dataset')
        shutil.rmtree(resolved)


@contextmanager
def lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'pipeline.lock'
    with path.open('x', encoding='utf-8') as stream:
        stream.write('Only one pipeline may use this state root.\n')
    try:
        yield
    finally:
        path.unlink()


def run_stage(command, log, cwd=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'Stage exited {result.returncode}; see {log}')


def validate_annotation(work):
    episodes = sorted((work / 'data').glob('chunk-*/episode_*'))
    if not episodes:
        raise ValueError('No dataset episodes')
    if len({p.name for p in episodes}) != len(episodes):
        raise ValueError('Repeated episode names across chunks')
    for episode in episodes:
        path = work / 'annotation' / (episode.name + '_structured.json')
        data = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(data, dict) or data.get('schema_version') != '1.0'
                or 'extraction' in data or 'scene_classification' not in data
                or 'task_name' not in data or not isinstance(data.get('subtask'), list)):
            raise ValueError(f'Incomplete structured annotation: {path}')
        for side in ('left', 'right'):
            feature = f'observation.images.head_{side}_rectified'
            if not (work / 'videos' / episode.parent.name / feature / (episode.name + '.mp4')).is_file():
                raise ValueError('Missing rectified video')
    if (work / 'annotation/argus').exists():
        raise ValueError('Unexpected Argus artifacts inside dataset')


def process(source, batch, args, client, stage='all'):
    name = source.rstrip('/').rsplit('/', 1)[-1]
    destination = prefix(args.destination) + batch + '/' + name + '/'
    work = args.local_root / batch / name
    state_path = args.state_root / batch / 'records' / (name + '.json')
    identity = dict(source=source, destination=destination, local_path=str(work))
    record = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else dict(identity, status='pending')
    if any(record.get(k) != v for k, v in identity.items()):
        raise ValueError('Source/destination/work path changed; use a new state root')
    if record.get('error'):
        return record  # Failed items are never automatically retried.
    def save(status, **extra):
        record.update(status=status, **extra)
        record.pop('error', None)
        atomic_json(state_path, record)
    logs = args.state_root / batch / 'logs' / name
    try:
        if record['status'] == 'complete':
            return record
        if stage == 'download' and record['status'] not in ('pending', 'downloading'):
            return record
        if stage in ('rectify', 'annotate') and record['status'] in ('annotated', 'uploaded'):
            return record
        if stage in ('upload', 'cleanup') and record['status'] not in ('annotated', 'uploaded'):
            raise ValueError('Previous stage is not complete')
        if stage in ('rectify', 'annotate') and record['status'] not in ('downloaded', 'rectified'):
            raise ValueError('Download is not complete')
        if stage == 'cleanup' and record['status'] != 'uploaded':
            raise ValueError('Upload is not complete')
        if record['status'] != 'uploaded':
            # Download only into a directory owned by this workflow.
            if record['status'] == 'pending' and work.exists():
                raise ValueError(f'Existing unowned dataset directory: {work}')
            if record['status'] == 'pending':
                save('downloading')
            if record['status'] == 'downloading':
                work.mkdir(parents=True, exist_ok=True)
                batch_csv = logs / 'download.csv'
                batch_csv.parent.mkdir(parents=True, exist_ok=True)
                with batch_csv.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['\u76ee\u6807\u8def\u5f84'])
                    writer.writerow([source])
                run_stage([args.python, '-m', 'roboego_hand_vis.download_data.base.run',
                           '--batch-csv', str(batch_csv), '--ks3-tool', str(args.ks3_tool),
                           '--ks3-config', str(args.ks3_config), '--output-dir', str(work.parent)],
                          logs / 'download.log', Path(__file__).resolve().parents[2])
                if not (work / 'meta/info.json').is_file():
                    raise ValueError('Downloaded dataset is missing meta/info.json')
                save('downloaded')
            if stage == 'download':
                return record
            if stage == 'annotate' and record['status'] == 'downloaded':
                raise ValueError('Rectification is not complete')
            if record['status'] == 'downloaded':
                run_stage([args.python, '-m', 'roboego_hand_vis.lerobot_stereo_rectification.jianzhi.run',
                           '--source', str(work), '--workers', '1', '--ffmpeg', args.ffmpeg],
                          logs / 'rectify.log', Path(__file__).resolve().parents[2])
                save('rectified')
            if stage == 'rectify':
                return record
            if record['status'] == 'rectified':
                run_stage(annotation_command(args, work),
                          logs / 'annotation.log', args.argus_repo)
                validate_annotation(work)
                build_readme(args, work, logs)
                save('annotated', readme_generated=True)
            if stage == 'annotate':
                return record
            if record['status'] != 'annotated':
                raise ValueError('Unexpected pipeline state')
            relocate_runtime_files(work, logs / 'runtime_artifacts')
            validate_annotation(work)
            if not record.get('readme_generated') or not (work / 'meta/readme.json').is_file():
                build_readme(args, work, logs)
                save('annotated', readme_generated=True)
            receipt = client.upload_directory(work, destination, recursive_copy=True)
            # Durable upload receipt must precede deletion.
            save('uploaded', upload=receipt)
        if stage == 'upload':
            return record
        files = record['upload']['file_sizes']
        verify_remote(client, destination, files)
        cleanup(work, args.local_root, files)
        save('complete', local_cleaned=True)
    except Exception as exc:
        # Retain the last successful stage so reruns do not repeat model calls.
        record['error'] = f'{type(exc).__name__}: {exc}'
        atomic_json(state_path, record)
    return record


def stage_worker(source, batch, args, stage):
    return process(source, batch, args, KS3(args.ks3_tool, getattr(args, 'upload_ks3_config', args.ks3_config)), stage)


def run_batch(batch, sources, args):
    skipped_path = args.state_root / batch / 'skipped.json'
    skipped = json.loads(skipped_path.read_text(encoding='utf-8')) if skipped_path.exists() else {}
    active = [source for source in sources if source not in skipped]
    stages = [('download', 0, {'downloaded', 'rectified', 'annotated', 'uploaded', 'complete'}),
              ('rectify', args.rectify_workers, {'rectified', 'annotated', 'uploaded', 'complete'}),
              ('annotate', args.annotation_workers, {'annotated', 'uploaded', 'complete'}),
              ('upload', 0, {'uploaded', 'complete'}),
              ('cleanup', 0, {'complete'})]
    for stage, workers, expected in stages:
        if not active:
            break
        print(json.dumps(dict(batch=batch, stage=stage, workers=workers, total=len(active))), flush=True)
        failures = 0
        succeeded = set()
        from contextlib import nullcontext
        from concurrent.futures import Future
        with (ProcessPoolExecutor(max_workers=workers) if workers else nullcontext()) as pool:
            if pool is None:
                futures = {}
                # One base downloader at a time; its own default jobs=64 controls transfers.
                for source in active:
                    future = Future()
                    try:
                        future.set_result(stage_worker(source, batch, args, stage))
                    except Exception as exc:
                        future.set_exception(exc)
                    futures[future] = source
            else:
                futures = {pool.submit(stage_worker, source, batch, args, stage): source for source in active}
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as exc:
                    record = dict(source=futures[future], error=str(exc), status='failed')
                source = futures[future]
                if record['status'] not in expected or record.get('error'):
                    failures += 1
                    skipped[source] = dict(stage=stage, status=record['status'],
                                           error=record.get('error', 'Stage did not complete'))
                    atomic_json(skipped_path, skipped)
                else:
                    succeeded.add(source)
                print(json.dumps(dict(batch=batch, stage=stage, result=record), ensure_ascii=False), flush=True)
        active = [source for source in active if source in succeeded]
        if failures:
            print(json.dumps(dict(batch=batch, stage=stage, failed=failures,
                                  message='Failed items skipped permanently; continue successful items')), flush=True)
    folder = args.local_root / batch
    if folder.is_dir() and not folder.is_symlink() and not any(folder.iterdir()):
        folder.rmdir()
    return not skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--batch-dir', type=Path, default=BASE/'data/jianzhi_batches')
    selection.add_argument('--batch-csv', type=Path)
    parser.add_argument('--session-id', help='Process only this 32-character session ID from the CSVs')
    parser.add_argument('--local-root', type=Path, default=Path('/home/jianzhi'))
    parser.add_argument('--state-root', type=Path, default=BASE/'data/jianzhi_pipeline_state')
    parser.add_argument('--argus-repo', type=Path, default=BASE/'code/argus')
    parser.add_argument('--ks3-tool', type=Path, default=BASE/'utils/ks3/ks3util-linux-amd64')
    parser.add_argument('--ks3-config', type=Path, default=BASE/'utils/ks3/ks3util-config-team')
    parser.add_argument('--upload-ks3-config', type=Path, default=BASE/'utils/ks3/ks3util-config-team')
    parser.add_argument('--destination', default=DESTINATION)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--ffmpeg', default=str(BASE/'envs/ego/bin/ffmpeg'))
    parser.add_argument('--model', default='qwen3.7-plus')
    parser.add_argument('--rectify-workers', type=int, default=8)
    parser.add_argument('--annotation-workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.session_id and not re.fullmatch(r'[0-9a-f]{32}', args.session_id):
        parser.error('session-id must be a 32-character lowercase hexadecimal ID')
    if min(args.rectify_workers, args.annotation_workers) < 1:
        parser.error('Worker counts must be positive')
    for key in ('local_root', 'state_root', 'argus_repo', 'ks3_tool', 'ks3_config', 'upload_ks3_config'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if (args.state_root.is_relative_to(args.local_root)
            or args.local_root.is_relative_to(args.state_root)):
        parser.error('State root and local root must be separate directories')
    if not args.model.startswith(('qwen', 'dashscope/')):
        parser.error('Structured annotation requires a Qwen model')
    if prefix(args.destination) != DESTINATION:
        parser.error('This workflow uploads only to ' + DESTINATION)
    batches = [args.batch_csv] if args.batch_csv else sorted(args.batch_dir.glob('batch_*.csv'))
    items = []
    seen = set()
    for csv_path in batches:
        if not re.fullmatch(r'batch_\d+', csv_path.stem):
            parser.error('Invalid batch filename')
        for source in read_batch(csv_path):
            name = source.rstrip('/').rsplit('/', 1)[-1]
            if (not source.startswith('ks3://baai-eai-pwm/jianzhi/')
                    or not re.fullmatch(r'[0-9a-f]{32}', name) or name in seen):
                parser.error('Invalid or duplicate Jianzhi session: ' + source)
            seen.add(name)
            if args.session_id and name != args.session_id:
                continue
            items.append((csv_path.stem, source))
    if not items:
        parser.error('No batch CSV entries found')
    if args.dry_run:
        for batch, source in items:
            name = source.rstrip('/').rsplit('/', 1)[-1]
            print(json.dumps(dict(source=source, local=str(args.local_root/batch/name),
                                 destination=DESTINATION+batch+'/'+name+'/', model=args.model,
                                 annotation_mode='existing_text_only', delete_after_verified_upload=True)))
        return 0
    for path in (args.ks3_tool, args.ks3_config, args.upload_ks3_config, args.argus_repo/'review/jianzhi/structure_existing.py'):
        if not path.is_file():
            parser.error('Missing required file: ' + str(path))
    grouped = {}
    for batch, source in items:
        grouped.setdefault(batch, []).append(source)
    with lock(args.state_root):
        failed_batches = 0
        for batch, sources in grouped.items():
            if not run_batch(batch, sources, args):
                failed_batches += 1
    print(json.dumps(dict(batches=len(grouped), batches_with_skipped_items=failed_batches)), flush=True)
    return int(bool(failed_batches))


if __name__ == '__main__':
    raise SystemExit(main())
