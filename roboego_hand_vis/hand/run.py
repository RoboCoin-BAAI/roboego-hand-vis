"""Thin native launcher: preserve the existing two algorithm commands."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roboego_hand_vis.hand import outputs
from roboego_hand_vis.common.processes import group_has_live_processes, stop_process_group
from roboego_hand_vis.hand.state import (
    SCHEMA_VERSION, atomic_json, checked_path, digest_file, fingerprint, load_state,
)


def build_plan(config, input_dir, output_dir, *, full_output=False):
    paths = config['paths']
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    mint = Path(paths['mint_root']).expanduser().resolve()
    ego = Path(paths['egohandkit_root']).expanduser().resolve()
    # Preserve venv executable symlinks: resolving them can select the base Python.
    mint_python = Path(paths['mint_python']).expanduser().absolute()
    ego_python = Path(paths['egohandkit_python']).expanduser().absolute()
    checkpoint = Path(paths['mint_checkpoint']).expanduser()
    checkpoint = (mint / checkpoint).resolve()
    video = (source / config.get('input', {}).get('video', 'videos/left_rectified.mp4')).resolve()
    depth = (source / config.get('input', {}).get('depth', 'depth')).resolve()
    work = output / '.work/attempts'
    prediction = work / 'mint-0001/prediction.npz'
    return {
        'input_dir': str(source), 'output_dir': str(output),
        'video': str(video), 'depth': str(depth), 'checkpoint': str(checkpoint),
        'prediction': str(prediction),
        'output_mode': 'full' if full_output else 'compact',
        'environment': {'EGOHANDKIT_ROOT': str(ego), 'MINT_ROOT': str(mint),
                        'EGOHANDKIT_PYTHON': str(ego_python)},
        'stages': [
            {'name': 'mint', 'cwd': str(mint), 'command': [
                str(mint_python), '-m', 'mint', 'infer', '--input', str(video),
                '--checkpoint', str(checkpoint), '--output', str(prediction.parent),
                '--camera-mode', 'chunked', '--max-frames', '10000', '--target-fps', '30',
                '--window', '32', '--window-batch-size', '1', '--no-render']},
            {'name': 'egohandkit', 'cwd': str(ego), 'command': [
                'bash', str(ego / 'scripts/run_without_front.sh'), 'mint', str(video),
                str(prediction), str(depth), str(work / 'egohandkit-0001')]},
        ],
    }


def preflight(plan, *, resume=False):
    output = Path(plan['output_dir'])
    if output.exists() and not resume:
        raise FileExistsError(f'Use a new output directory: {output}')
    env = plan['environment']
    mint, ego = Path(env['MINT_ROOT']), Path(env['EGOHANDKIT_ROOT'])
    required = [Path(plan['video']), Path(plan['checkpoint']), ego / 'run.py',
                ego / 'scripts/run_without_front.sh', ego / 'scripts/export_egohandkit_cache_mint.py']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (mint / 'mint', mint / 'scripts', Path(plan['depth'])):
        if not path.is_dir():
            raise NotADirectoryError(path)
    for path in (plan['stages'][0]['command'][0], env['EGOHANDKIT_PYTHON']):
        if not Path(path).is_file() or not os.access(path, os.X_OK):
            raise ValueError(f'Python executable unavailable: {path}')
    if 'EGOHANDKIT_PYTHON' not in (ego / 'scripts/run_without_front.sh').read_text():
        raise ValueError('EgoHandKit launcher needs the native path/Python configuration patch')
    exporter = mint / 'scripts/export_egohandkit_cache.py'
    if exporter.exists() and exporter.read_bytes() != (ego / 'scripts/export_egohandkit_cache_mint.py').read_bytes():
        raise ValueError('MINT exporter differs from EgoHandKit; reconcile before running')


def revision(root):
    def git(*args):
        try:
            result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    return {'commit': git('rev-parse', 'HEAD'), 'changes': git('status', '--porcelain')}


def run_stage(stage, environment, log_path, *, lock_fds=()):
    with log_path.open('a') as log:
        process = subprocess.Popen(stage['command'], cwd=stage['cwd'], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=lock_fds)
        try:
            code = process.wait()
            if sys.platform.startswith("linux") and group_has_live_processes(process.pid):
                stop_process_group(process)
            return code
        except BaseException:
            if sys.platform.startswith("linux"):
                stop_process_group(process)
            elif process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def log_event(log_path, event, **fields):
    record = {'time': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
    with log_path.open('a') as log:
        log.write('\n[launcher] ' + json.dumps(record, ensure_ascii=False) + '\n')
        log.flush()
        os.fsync(log.fileno())


def validate(plan, kind, artifact, log, lock_fds):
    output = Path(plan['output_dir'])
    result = checked_path(output, f'.work/validation-{uuid.uuid4().hex}.json')
    environment = {**os.environ, **plan['environment'], 'PYTHONUNBUFFERED': '1'}
    stage = {'name': f'validate_{kind}', 'cwd': plan['input_dir'], 'command': [
        environment['EGOHANDKIT_PYTHON'], str(Path(outputs.__file__).resolve()),
        kind, str(artifact), plan['video'], str(result)]}
    log_event(log, 'validation_started', command=stage['command'])
    code = run_stage(stage, environment, log, lock_fds=lock_fds)
    if code:
        raise RuntimeError(f'{kind} artifact validation failed (exit {code}); see run.log')
    metadata = json.loads(result.read_text())
    result.unlink()
    return metadata


def artifact_hashes(output, artifact, kind):
    names = [artifact] if kind == 'mint' else [artifact / name for name in outputs.DELIVERY_NAMES]
    result = {}
    for path in names:
        path = checked_path(output, path.relative_to(output))
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing or empty artifact: {path}')
        result[str(path.relative_to(output))] = digest_file(path)
    return result


def stage_artifact(output, attempt, name):
    if name == 'mint':
        return checked_path(output, attempt.relative_to(output) / 'prediction.npz')
    mint = checked_path(output, attempt.relative_to(output) / 'mint')
    candidates = list(mint.glob('*/final/hand_tracking.parquet'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one final hand tracking file, found {len(candidates)}')
    final = checked_path(output, candidates[0].parent.relative_to(output))
    checked_path(output, (final / 'summary.json').relative_to(output))
    return final


def execute(plan, *, resume=False):
    output = Path(plan['output_dir'])
    # Serialize both shared EgoHandKit caches and this output root. Children inherit
    # the locks so killing only the launcher cannot start a competing resumed run.
    key = hashlib.sha256(plan['environment']['EGOHANDKIT_ROOT'].encode()).hexdigest()[:24]
    lock_path = Path(tempfile.gettempdir()) / f'roboego-native-hand-{os.getuid()}-{key}.lock'
    output_key = hashlib.sha256(str(output).encode()).hexdigest()[:24]
    output_lock = Path(tempfile.gettempdir()) / f'roboego-hand-output-{os.getuid()}-{output_key}.lock'
    with ExitStack() as stack:
        locks = []
        for path in (lock_path, output_lock):
            lock = stack.enter_context(path.open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another native hand launch is using this EgoHandKit checkout or output')
            locks.append(lock.fileno())
        lock_fds = tuple(locks)
        work = checked_path(output, '.work')
        state_path = checked_path(output, '.work/state.json')
        log = checked_path(output, 'run.log')
        if resume:
            if not state_path.is_file():
                raise ValueError('No resumable checkpoint: output is already compacted or was not created by this launcher')
            try:
                state = load_state(state_path)
                if state['plan'] != plan:
                    raise ValueError('Resume plan differs (including output mode); use original arguments or a new output')
                if not state.get('delivery'):
                    preflight(plan, resume=True)
                    if state['fingerprint'] != fingerprint(plan):
                        raise ValueError('Inputs, code, model or runtime changed; use a new output directory')
            except Exception as error:
                log_event(log, 'resume_refused', error=str(error))
                raise
        else:
            preflight(plan)
            signature = fingerprint(plan)
            output.mkdir(parents=True, exist_ok=False)
            work.mkdir()
            state = {'schema_version': SCHEMA_VERSION, 'plan': plan, 'fingerprint': signature,
                     'stages': {}, 'delivery': None}
            atomic_json(state_path, state)
        log_event(log, 'resume' if resume else 'start', run_id=uuid.uuid4().hex, plan=plan,
                  fingerprint=state['fingerprint'], sources={name: revision(path) for name, path in {
                      'project': Path(__file__).resolve().parents[2],
                      'mint': plan['environment']['MINT_ROOT'],
                      'egohandkit': plan['environment']['EGOHANDKIT_ROOT']}.items()})
        environment = {**os.environ, **plan['environment'], 'PYTHONUNBUFFERED': '1'}
        started = time.monotonic()
        code = 1
        try:
            if not state.get('delivery'):
                for original in plan['stages']:
                    name = original['name']
                    previous = state['stages'].get(name, {})
                    if previous.get('status') == 'complete':
                        artifact = checked_path(output, previous['artifact'])
                        if artifact_hashes(output, artifact, name) != previous['hashes']:
                            raise ValueError(f'Completed {name} artifact changed; use a new output directory')
                        log_event(log, 'stage_skipped', stage=name, reason='completed checkpoint and matching artifact hashes')
                        continue
                    attempts = checked_path(output, '.work/attempts')
                    attempts.mkdir(exist_ok=True)
                    number = previous.get('attempt_number', 0) + 1
                    attempt = checked_path(output, f'.work/attempts/{name}-{number:04d}')
                    if attempt.exists():
                        raise FileExistsError(f'Attempt already exists: {attempt}')
                    stage = {**original, 'command': list(original['command'])}
                    if name == 'mint':
                        stage['command'][stage['command'].index('--output') + 1] = str(attempt)
                    else:
                        stage['command'][4] = str(checked_path(output, state['stages']['mint']['artifact']))
                        stage['command'][-1] = str(attempt)
                    state['stages'][name] = {'status': 'running', 'attempt_number': number,
                                             'command': stage['command']}
                    atomic_json(state_path, state)
                    log_event(log, 'stage_started', stage=name, attempt=number, cwd=stage['cwd'], command=stage['command'])
                    print(f'Running {name}; log: {log}', flush=True)
                    before = time.monotonic()
                    result = run_stage(stage, environment, log, lock_fds=lock_fds)
                    log_event(log, 'stage_exited', stage=name, returncode=result, wall_s=time.monotonic() - before)
                    if result:
                        code = result if result > 0 else 128 - result
                        raise RuntimeError(f'{name} failed: exit {result}')
                    artifact = stage_artifact(output, attempt, name)
                    hashes = artifact_hashes(output, artifact, name)
                    metadata = validate(plan, 'mint' if name == 'mint' else 'final', artifact, log, lock_fds)
                    state['stages'][name].update(status='complete', artifact=str(artifact.relative_to(output)),
                                                hashes=hashes, validation=metadata)
                    atomic_json(state_path, state)
                    log_event(log, 'stage_completed', stage=name, validation=metadata)
                final = checked_path(output, state['stages']['egohandkit']['artifact'])
                state['delivery'] = outputs.publish(final, output)
                atomic_json(state_path, state)
                log_event(log, 'published', hashes=state['delivery'],
                          validation=state['stages']['egohandkit']['validation'])
            else:
                log_event(log, 'delivery_recovery', reason='publication already checkpointed; no inference needed')
            outputs.verify_delivery(output, state['delivery'])
            if plan['output_mode'] == 'compact':
                log_event(log, 'cleanup_started', path=str(work))
                outputs.cleanup_work(output)
                log_event(log, 'cleanup_completed')
            log_event(log, 'complete', output_mode=plan['output_mode'], quality_status='not_validated')
            code = 0
        except KeyboardInterrupt:
            code = 130
            log_event(log, 'interrupted', message='Resume explicitly with --resume; intermediates retained')
        except Exception as error:
            log_event(log, 'failed', error=str(error), message='Available intermediates retained; see checkpoint')
        finally:
            log_event(log, 'run_finished', returncode=code, wall_s=time.monotonic() - started)
        print(f"{'complete' if code == 0 else 'failed'}: {log}", flush=True)
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('recommended.yaml'))
    parser.add_argument('--dry-run', action='store_true', help='Print commands only; no execution or output creation')
    parser.add_argument('--full-output', action='store_true', help='Keep .work intermediates after successful delivery')
    parser.add_argument('--resume', action='store_true', help='Resume a matching checkpoint; incomplete stages restart')
    args = parser.parse_args(argv)
    with args.config.expanduser().open() as stream:
        config = yaml.safe_load(stream)
    plan = build_plan(config, args.input_dir.expanduser(), args.output_dir.expanduser(), full_output=args.full_output)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        for stage in plan['stages']:
            print(f"[{stage['name']}] cwd={stage['cwd']}\n{shlex.join(stage['command'])}")
        return 0
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return execute(plan, resume=args.resume)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
