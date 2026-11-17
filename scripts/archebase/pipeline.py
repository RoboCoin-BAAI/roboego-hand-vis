"""Download raw MCAP -> convert -> undistort hands/images -> model clean -> upload -> cleanup."""
import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
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

DESTINATION = 'ks3://baai-eai-pwm/archebase_lerobot/'
BASE = Path('/share/project/liuyou/humancoin')


def annotation_command(args, dataset, output):
    return [args.python, '-m', 'review.archebase.run',
            '--input', str(dataset/'annotation/episode_000000_source.json'),
            '--out', str(output), '--model', args.model]


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


def cleanup(work, local_root, expected_files):
    base = Path(local_root).resolve()
    work = Path(work)
    resolved = work.resolve()
    if (work.is_symlink() or resolved == base or not resolved.is_relative_to(base)
            or len(resolved.relative_to(base).parts) != 2
            or not re.fullmatch(r'batch_\d+', resolved.parent.name)
            or not re.fullmatch(r'[0-9]+', resolved.name)):
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


def validate_annotation(dataset):
    data = json.loads((dataset/'annotation/episode_000000_structured.json').read_text(encoding='utf-8'))
    prepared = json.loads((dataset/'annotation/episode_000000_source.json').read_text(encoding='utf-8'))
    expected = [r for r in prepared['segments'] if '\u65e0\u6548' not in r['source_segment'].get('verbs', [])]
    if (data.get('schema_version') != '1.0' or 'scene_classification' not in data
            or 'task_name' not in data or not isinstance(data.get('subtask'), list)
            or len(data['subtask']) != len(expected)):
        raise ValueError('Incomplete model-cleaned annotation')
    for row, original in zip(data['subtask'], expected):
        if (row['start_s'],row['end_s']) != (original['start_s'],original['end_s']):
            raise ValueError('Annotation times differ from source')
    if re.search(r'[\u3400-\u9fff]', json.dumps(data.get('subtask'), ensure_ascii=False)):
        raise ValueError('Untranslated Chinese in final subtasks')
    validate_undistortion(dataset)


def validate_undistortion(dataset):
    info = json.loads((dataset/'meta/info.json').read_text(encoding='utf-8'))
    manifest = json.loads((dataset/'calibrations/archebase_undistortion.json').read_text(encoding='utf-8'))
    episode = dataset/'data/chunk-000/episode_000000'
    for side in ('left','right'):
        key = f'observation.images.head_{side}_undistorted'
        if (key not in info['features'] or not (episode/f'{key}.parquet').is_file()
                or not (dataset/f'videos/chunk-000/{key}/episode_000000.mp4').is_file()
                or manifest['frame_counts'].get(f'episode_000000/head_{side}',0)<=0):
            raise ValueError('Missing undistorted camera feature')
        for suffix in ('points_undistorted','points_2d_undistorted'):
            key = f'observation.hand_{side}_{suffix}'
            if key not in info['features'] or not (episode/f'{key}.parquet').is_file():
                raise ValueError('Missing transformed hand points')


def publish_annotation(dataset, output):
    receipt = json.loads(output.with_name('.'+output.name+'.claim.json').read_text(encoding='utf-8'))
    if (receipt.get('status') != 'complete'
            or receipt.get('output_sha256') != hashlib.sha256(output.read_bytes()).hexdigest()):
        raise ValueError('Model-cleaning receipt does not match result')
    final = dataset/'annotation/episode_000000_structured.json'
    final.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(output,final)
    data = json.loads(final.read_text(encoding='utf-8'))
    task = data.get('task_name') or data.get('task_description')
    if task:
        for name in ('tasks.jsonl','episodes.jsonl'):
            path = dataset/'meta'/name
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            for row in rows:
                if name == 'tasks.jsonl': row['task'] = task
                else: row['tasks'] = [task]
            temporary = path.with_suffix('.jsonl.tmp')
            temporary.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
            temporary.replace(path)


def process(source, batch, args, client, stage='all'):
    name = source.rstrip('/').rsplit('/', 1)[-1]
    if (not source.startswith('ks3://baai-eai-pwm/archebase_1000/')
            or not re.fullmatch(r'[0-9]+',name) or not re.fullmatch(r'batch_\d+',batch)):
        raise ValueError('Invalid Archebase source/batch identity')
    destination = prefix(args.destination) + batch + '/' + name + '/'
    work = args.local_root / batch / name
    state_path = args.state_root / batch / 'records' / (name + '.json')
    identity = dict(source=source, destination=destination, local_path=str(work))
    configuration = {key:str(getattr(args,key,'')) for key in ('model','balance','converter_repo','argus_repo','python','ffmpeg')}
    identity['configuration'] = configuration
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
        if stage == 'prepare' and record['status'] in ('annotated', 'uploaded'):
            return record
        if stage in ('upload', 'cleanup') and record['status'] not in ('annotated', 'uploaded'):
            raise ValueError('Previous stage is not complete')
        if stage == 'prepare' and record['status'] not in ('downloaded', 'converted', 'rectified', 'annotating'):
            raise ValueError('Download is not complete')
        if stage == 'cleanup' and record['status'] != 'uploaded':
            raise ValueError('Upload is not complete')
        dataset = work / 'dataset'
        raw = work / 'raw'
        if record['status'] != 'uploaded':
            if record['status'] == 'pending' and work.exists():
                raise ValueError(f'Existing unowned local directory: {work}')
            if record['status'] == 'pending':
                save('downloading')
            if record['status'] == 'downloading':
                raw.mkdir(parents=True, exist_ok=True)
                objects = {item['uri']: item for item in client.objects(source)}
                for filename in ('data.mcap', 'annotation.json'):
                    uri = source.rstrip('/')+'/'+filename
                    item = objects.get(uri)
                    if not item or item['size'] <= 0:
                        raise ValueError('Missing/nonempty raw source object: '+uri)
                    client.download(dict(item,source_uri=uri), raw/filename)
                json.loads((raw/'annotation.json').read_text(encoding='utf-8'))
                save('downloaded')
            if stage == 'download':
                return record
            if record['status'] == 'downloaded':
                if not dataset.exists():
                    run_stage([args.python, str(args.converter_repo/'format_convert/archebase/convert_archebase_mcap.py'),
                               '--source', str(raw), '--out', str(dataset), '--task', name, '--ffmpeg', args.ffmpeg],
                              logs/'convert.log', args.converter_repo)
                meta = json.loads((dataset/'meta/archebase_source.json').read_text(encoding='utf-8'))
                with (raw/'data.mcap').open('rb') as stream:
                    source_hash = hashlib.file_digest(stream,'sha256').hexdigest()
                if meta['source_sha256'] != source_hash:
                    raise ValueError('Converted dataset source hash mismatch')
                if not (dataset/'annotation/episode_000000_source.json').is_file():
                    raise ValueError('Conversion did not produce intermediate annotations')
                save('converted')
            if record['status'] == 'converted':
                if not (dataset/'calibrations/archebase_undistortion.json').exists():
                    run_stage([args.python, '-m', 'roboego_hand_vis.lerobot_stereo_rectification.archebase.run',
                               '--source', str(dataset), '--balance', str(args.balance), '--ffmpeg', args.ffmpeg],
                              logs/'undistort.log', Path(__file__).resolve().parents[2])
                validate_undistortion(dataset)
                save('rectified')
            if record['status'] in ('rectified','annotating'):
                output = logs/'model/episode_000000_structured.json'
                save('annotating')
                run_stage(annotation_command(args,dataset,output),logs/'annotation.log',args.argus_repo)
                publish_annotation(dataset,output)
                validate_annotation(dataset)
                save('annotated')
            if stage == 'prepare':
                return record
            if record['status'] != 'annotated':
                raise ValueError('Unexpected pipeline state')
            validate_annotation(dataset)
            receipt = client.upload_directory(dataset,destination)
            # Source raw files are not uploaded, but cleanup covers both trees.
            save('uploaded',upload=receipt,cleanup_file_sizes=file_sizes(work))
        if stage == 'upload':
            return record
        files = record['upload']['file_sizes']
        verify_remote(client, destination, files)
        cleanup(work, args.local_root, record['cleanup_file_sizes'])
        save('complete', local_cleaned=True)
    except Exception as exc:
        # Retain the last successful stage so reruns do not repeat model calls.
        record['error'] = f'{type(exc).__name__}: {exc}'
        atomic_json(state_path, record)
    return record


def stage_worker(source, batch, args, stage):
    return process(source, batch, args, KS3(args.ks3_tool, args.ks3_config), stage)


def run_batch(batch, sources, args):
    skipped_path = args.state_root / batch / 'skipped.json'
    skipped = json.loads(skipped_path.read_text(encoding='utf-8')) if skipped_path.exists() else {}
    active = [source for source in sources if source not in skipped]
    stages = [('download', args.download_workers, {'downloaded', 'converted', 'rectified', 'annotating', 'annotated', 'uploaded', 'complete'}),
              ('prepare', args.prepare_workers, {'annotated', 'uploaded', 'complete'}),
              ('upload', args.upload_workers, {'uploaded', 'complete'}),
              ('cleanup', args.upload_workers, {'complete'})]
    for stage, workers, expected in stages:
        if not active:
            break
        print(json.dumps(dict(batch=batch, stage=stage, workers=workers, total=len(active))), flush=True)
        failures = 0
        succeeded = set()
        with ProcessPoolExecutor(max_workers=workers) as pool:
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
    selection.add_argument('--batch-dir', type=Path, default=BASE/'data/archebase_batches')
    selection.add_argument('--batch-csv', type=Path)
    parser.add_argument('--session-id', help='Process only this numeric episode ID from the CSVs')
    parser.add_argument('--local-root', type=Path, default=Path('/home/archebase'))
    parser.add_argument('--state-root', type=Path, default=BASE/'data/archebase_pipeline_state')
    parser.add_argument('--argus-repo', type=Path, default=BASE/'code/argus')
    parser.add_argument('--ks3-tool', type=Path, default=BASE/'utils/ks3/ks3util-linux-amd64')
    parser.add_argument('--ks3-config', type=Path, default=BASE/'utils/ks3/ks3util-config')
    parser.add_argument('--destination', default=DESTINATION)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--ffmpeg', default=str(BASE/'envs/ego/bin/ffmpeg'))
    parser.add_argument('--model', default='qwen3.6-plus')
    parser.add_argument('--converter-repo', type=Path, default=BASE/'code/Multi-Frequency-LeRobot')
    parser.add_argument('--balance', type=float, default=0.)
    parser.add_argument('--download-workers', type=int, default=4)
    parser.add_argument('--prepare-workers', type=int, default=4)
    parser.add_argument('--upload-workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.session_id and not re.fullmatch(r'[0-9]+', args.session_id):
        parser.error('session-id must be a numeric ID')
    if min(args.download_workers, args.prepare_workers, args.upload_workers) < 1:
        parser.error('Worker counts must be positive')
    for key in ('local_root', 'state_root', 'argus_repo', 'ks3_tool', 'ks3_config', 'converter_repo'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if (args.state_root.is_relative_to(args.local_root)
            or args.local_root.is_relative_to(args.state_root)):
        parser.error('State root and local root must be separate directories')
    if not args.model.startswith(('qwen', 'dashscope/')):
        parser.error('Structured annotation requires a Qwen model')
    if not math.isfinite(args.balance) or not 0 <= args.balance <= 1:
        parser.error('balance must be between 0 and 1')
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
            if (not source.startswith('ks3://baai-eai-pwm/archebase_1000/')
                    or not re.fullmatch(r'[0-9]+', name) or name in seen):
                parser.error('Invalid or duplicate Archebase episode: ' + source)
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
                                 annotation_mode='source_text_cleaning', delete_after_verified_upload=True)))
        return 0
    for path in (args.ks3_tool, args.ks3_config, args.argus_repo/'review/archebase/run.py', args.converter_repo/'format_convert/archebase/convert_archebase_mcap.py'):
        if not path.is_file():
            parser.error('Missing required file: ' + str(path))
    grouped = {}
    for batch, source in items:
        grouped.setdefault(batch, []).append(source)
    with lock(args.state_root):
        # Fail before downloading a batch if the model route has no credentials.
        run_stage([args.python,'-c','import sys; from review.qwen_endpoint import settings; settings(sys.argv[1])',args.model],
                  args.state_root/'preflight.log',args.argus_repo)
        failed_batches = 0
        for batch, sources in grouped.items():
            if not run_batch(batch, sources, args):
                failed_batches += 1
    print(json.dumps(dict(batches=len(grouped), batches_with_skipped_items=failed_batches)), flush=True)
    return int(bool(failed_batches))


if __name__ == '__main__':
    raise SystemExit(main())
