"""Thin native launcher: preserve the existing two algorithm commands."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import time

import yaml


def build_plan(config, input_dir, output_dir):
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
    prediction = output / 'mint_pre/prediction.npz'
    return {
        'input_dir': str(source), 'output_dir': str(output),
        'video': str(video), 'depth': str(depth), 'checkpoint': str(checkpoint),
        'prediction': str(prediction),
        'environment': {'EGOHANDKIT_ROOT': str(ego), 'MINT_ROOT': str(mint),
                        'EGOHANDKIT_PYTHON': str(ego_python)},
        'stages': [
            {'name': 'mint', 'cwd': str(mint), 'command': [
                str(mint_python), '-m', 'mint', 'infer', '--input', str(video),
                '--checkpoint', str(checkpoint), '--output', str(output / 'mint_pre'),
                '--camera-mode', 'chunked', '--max-frames', '10000', '--target-fps', '30',
                '--window', '32', '--window-batch-size', '1', '--no-render']},
            {'name': 'egohandkit', 'cwd': str(ego), 'command': [
                'bash', str(ego / 'scripts/run_without_front.sh'), 'mint', str(video),
                str(prediction), str(depth), str(output)]},
        ],
    }


def preflight(plan):
    output = Path(plan['output_dir'])
    if output.exists():
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


def run_stage(stage, environment, log_path):
    with log_path.open('x') as log:
        process = subprocess.Popen(stage['command'], cwd=stage['cwd'], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait()
        except BaseException:
            if process.poll() is None:
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


def execute(plan):
    preflight(plan)
    output = Path(plan['output_dir'])
    # Existing EgoHandKit caches live under its source tree. Serialize this launcher
    # per checkout rather than pretending separate output roots isolate that cache.
    key = hashlib.sha256(plan['environment']['EGOHANDKIT_ROOT'].encode()).hexdigest()[:24]
    lock_path = Path(tempfile.gettempdir()) / f'roboego-native-hand-{os.getuid()}-{key}.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another native hand launch is using this EgoHandKit checkout')
        output.mkdir(parents=True, exist_ok=False)
        report = {**plan, 'status': 'failed', 'quality_status': 'not_validated',
                  'sources': {name: revision(path) for name, path in {
                      'project': Path(__file__).resolve().parents[2],
                      'mint': plan['environment']['MINT_ROOT'],
                      'egohandkit': plan['environment']['EGOHANDKIT_ROOT']}.items()},
                  'measurements': []}
        environment = {**os.environ, **plan['environment']}
        started = time.monotonic()
        code = 1
        try:
            for stage in plan['stages']:
                log_path = output / f"{stage['name']}.log"
                print(f"Running {stage['name']}; log: {log_path}", flush=True)
                before = time.monotonic()
                result = run_stage(stage, environment, log_path)
                report['measurements'].append({'stage': stage['name'], 'returncode': result,
                                               'wall_s': time.monotonic() - before})
                if result:
                    code = result if result > 0 else 128 - result
                    raise RuntimeError(f"{stage['name']} failed: exit {result}; see {log_path}")
                if stage['name'] == 'mint':
                    prediction = Path(plan['prediction'])
                    if not prediction.is_file() or prediction.stat().st_size == 0:
                        raise RuntimeError('MINT exited successfully but prediction.npz is missing or empty')
            report['status'], code = 'complete', 0
        except KeyboardInterrupt:
            report['status'], code = 'interrupted', 130
        except Exception as error:
            report['error'] = str(error)
        finally:
            report['wall_s'] = time.monotonic() - started
            report['returncode'] = code
            # Output compaction/publication can be added here later. Retain everything now.
            (output / 'run_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f"{report['status']}: {output / 'run_report.json'}", flush=True)
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('recommended.yaml'))
    parser.add_argument('--dry-run', action='store_true', help='Print commands only; no execution or output creation')
    args = parser.parse_args(argv)
    with args.config.expanduser().open() as stream:
        config = yaml.safe_load(stream)
    plan = build_plan(config, args.input_dir.expanduser(), args.output_dir.expanduser())
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        for stage in plan['stages']:
            print(f"[{stage['name']}] cwd={stage['cwd']}\n{shlex.join(stage['command'])}")
        return 0
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return execute(plan)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
