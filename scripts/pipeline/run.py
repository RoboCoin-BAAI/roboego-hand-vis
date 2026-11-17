"""Shared-filesystem CPU/GPU stage runner. No database or shell command interpolation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
from uuid import uuid4


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '-' + uuid4().hex)
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def cleanup_success_work(folder, stage):
    """Discard successful L3 work files, retaining only final data and stage state."""
    names = {'review': ['review_work'], 'rectify': ['rectify_work', 'encoder_logs'],
             'depth': ['depth_work']}.get(stage, [])
    for name in names:
        path = folder / name
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)


@contextmanager
def claim(path):
    """OS releases the lock on crash; never steal work based on elapsed time."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_config(config):
    if config.get('cpu_failure_policy','stop') not in ('stop','continue_passed'):
        raise ValueError('cpu_failure_policy must be stop or continue_passed')
    stages = config['stages']
    start = config.get('start_at', 'download')
    if start not in ('download', 'convert', 'qc'):
        raise ValueError('start_at must be download, convert or qc; GPU gates cannot be bypassed')
    if start != 'download':
        if start not in stages:
            raise ValueError('Unknown start_at stage')
        stages[start]['depends'] = []
        excluded = ['download'] + (['convert'] if start == 'qc' else [])
        config['run_stages'] = [n for n in config['run_stages'] if n not in excluded]
    visiting, visited = set(), set()
    def visit(name):
        if name in visiting:
            raise ValueError('Cyclic stage dependency: ' + name)
        if name in visited:
            return
        if name not in stages:
            raise ValueError('Unknown dependency: ' + name)
        visiting.add(name)
        stage = stages[name]
        if stage['role'] not in ('cpu', 'gpu'):
            raise ValueError('Invalid role: ' + name)
        for parent in stage.get('depends', []):
            visit(parent)
        visiting.remove(name)
        visited.add(name)
    for name in stages:
        visit(name)
    for key in ('run_stages', 'force_stages'):
        if not set(config.get(key, [])).issubset(stages):
            raise ValueError('Unknown stage in ' + key)
    if not set(config.get('force_stages', [])).issubset(config['run_stages']):
        raise ValueError('force_stages must be included in run_stages')


def tasks(config, batches):
    from urllib.parse import urlsplit
    if config.get('start_at') == 'qc':
        root = Path(config['paths']['output_root']).resolve()
        if any(not b.startswith('batch_') or not b[6:].isdigit() for b in batches):
            raise ValueError('Invalid batch selection')
        folders = [root / b for b in batches] if batches else sorted(root.glob('batch_*'))
        result = []
        for batch in folders:
            found = []
            if batch.is_dir():
                for info in sorted(batch.glob('*/meta/info.json')):
                    session = info.parent.parent.resolve()
                    if info.is_file() and not info.parent.parent.name.startswith('.') and session.is_relative_to(root):
                        found.append(dict(batch=batch.name, name=info.parent.parent.name, source=str(session)))
            if not found and batches:
                raise ValueError('No converted sessions with meta/info.json in ' + str(batch))
            result.extend(found)
        if not result:
            raise ValueError('No converted sessions with batch_*/session/meta/info.json in ' + str(root))
        return result
    root = Path(config['paths']['input_root'])
    files = [root / (b + '.csv') for b in batches] if batches else sorted(root.glob('batch_*.csv'))
    if not files:
        raise ValueError('No batch CSVs')
    result = []
    for file in files:
        if file.parent != root or not file.stem.startswith('batch_'):
            raise ValueError('Invalid batch: ' + str(file))
        with file.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        seen = set()
        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            source = row[0].strip()
            name = urlsplit(source).path.rstrip('/').rsplit('/', 1)[-1]
            if len(row) != 1 or not name or name in ('.', '..') or '\\' in name or name in seen:
                raise ValueError('Invalid/duplicate session in ' + str(file))
            seen.add(name)
            result.append(dict(batch=file.stem, name=name, source=source, csv=str(file)))
    return result


def rectified_index(config, batch):
    if not batch.startswith('batch_') or not batch[6:].isdigit():
        raise ValueError('Invalid batch: ' + batch)
    root = (Path(config['paths']['state_root']) / config['dataset']
            if config.get('level') == 'L3' else Path(config['paths']['output_root']))
    return root / batch / 'rectification_done.json'


def update_rectified_index(config, session, done):
    root = Path(config['paths']['output_root']).resolve()
    session = Path(session).resolve()
    if not session.is_relative_to(root) or len(session.relative_to(root).parts) != 2:
        raise ValueError('Invalid rectified session: ' + str(session))
    path = rectified_index(config, session.parent.name)
    while True:
        with claim(path.with_suffix('.lock')) as acquired:
            if acquired:
                paths = read(path) if path.exists() else []
                if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
                    raise ValueError('Invalid rectification index: ' + str(path))
                paths = set(paths)
                session = str(Path(session).resolve())
                if done:
                    paths.add(session)
                else:
                    paths.discard(session)
                write(path, sorted(paths))
                return
        time.sleep(.05)


def gpu_tasks(config, batches):
    root = Path(config['paths']['output_root']).resolve()
    indices = ([rectified_index(config, b) for b in dict.fromkeys(batches)] if batches
               else sorted((Path(config['paths']['state_root']) / config['dataset']
                            if config.get('level') == 'L3' else root).glob('batch_*/rectification_done.json')))
    paths = []
    for index in indices:
        if not index.exists():
            continue
        values = read(index)
        if not isinstance(values, list) or any(not isinstance(p, str) for p in values):
            raise ValueError('Invalid rectification index: ' + str(index))
        for value in values:
            session = Path(value).resolve()
            if session.parent != (root / index.parent.name).resolve():
                raise ValueError('Session does not belong to indexed batch: ' + value)
        paths.extend(values)
    selected = []
    for text in dict.fromkeys(paths):
        session = Path(text).resolve()
        if not session.is_relative_to(root) or len(session.relative_to(root).parts) != 2:
            raise ValueError('Invalid indexed session: ' + text)
        batch, name = session.relative_to(root).parts
        if batches and batch not in batches:
            continue
        state = read(Path(config['paths']['state_root']) / config['dataset'] / batch / name / 'state.json')
        record = state.get('rectify', {})
        if record.get('status') != 'done':
            continue
        task = record.get('task')
        if not isinstance(task, dict) or task.get('batch') != batch or task.get('name') != name:
            raise ValueError('Rectification state has no matching task; rerun CPU to register: ' + text)
        selected.append(task)
    return selected


class Runner:
    def __init__(self, config, role, run_id=None, retry=False, adopt=False):
        self.config, self.role = config, role
        self.run_id = run_id or uuid4().hex
        self.retry, self.adopt = retry, adopt
        self.registered_rectifications = set()

    def folder(self, task):
        return Path(self.config['paths']['state_root']) / self.config['dataset'] / task['batch'] / task['name']

    def values(self, task, gpu='0'):
        values = dict(self.config['paths'], **task, gpu=str(gpu), level=self.config['level'])
        values['session'] = str(Path(values['output_root']) / task['batch'] / task['name'])
        values['raw_session'] = str(Path(values['input_root']) / task['batch'] / task['name'])
        return values

    def command(self, spec, values):
        if not isinstance(spec, list) or not spec or any(not isinstance(s, str) for s in spec):
            raise ValueError('Commands must be nonempty argument lists')
        return [s.format_map(values) for s in spec]

    def signature(self, name, task, states):
        stage = self.config['stages'][name]
        values = self.values(task, gpu='{gpu}')
        files = []
        for pattern in stage.get('inputs', []):
            # Relative patterns are interpreted under the current session.
            for path in sorted(Path(values['session']).glob(pattern)):
                if path.is_file():
                    if any('_rectified' in part for part in path.parts):
                        continue
                    stat = path.stat()
                    files.append((str(path.relative_to(values['session'])), stat.st_size, stat.st_mtime_ns))
        return digest(dict(stage=stage, source=task['source'], files=files,
                           parents={p: states.get(p, {}).get('generation') for p in stage.get('depends', [])}))

    def current(self, name, task, states, active=None):
        active = active or set()
        if name in active:
            return False
        stage = self.config['stages'][name]
        record = states.get(name, {})
        return (record.get('status') == 'done' and record.get('signature') == self.signature(name, task, states)
                and record.get('artifacts', []) == self.artifacts(name, task)
                and all(self.current(p, task, states, active | {name}) for p in stage.get('depends', [])))

    def artifacts(self, name, task):
        root = Path(self.values(task)['session'])
        files = []
        for pattern in self.config['stages'][name].get('outputs', []):
            matches = sorted(root.glob(pattern))
            if not matches:
                return None
            for path in matches:
                if path.is_file():
                    stat = path.stat()
                    files.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
        return [list(f) for f in files]

    def invalidate(self, name, states):
        for child, stage in self.config['stages'].items():
            if name in stage.get('depends', []):
                if child in states:
                    states[child]['status'] = 'stale'
                self.invalidate(child, states)

    def step(self, task, gpu='0', stages=None):
        folder = self.folder(task)
        with claim(folder / 'session.lock') as acquired:
            if not acquired:
                return False
            states = read(folder / 'state.json')
            for old_name, old in states.items():
                if old.get('status') == 'running':
                    for path_text, backup_text in reversed(old.get('backups', [])):
                        path, backup = Path(path_text), Path(backup_text)
                        if backup.exists():
                            if path.exists():
                                retained = folder / 'interrupted' / uuid4().hex
                                retained.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(path), str(retained))
                            path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(backup), str(path))
                    old.update(status='failed', error='Previous runner interrupted; OS lock is now free')
                    write(folder / 'state.json', states)
            for name in (self.config['run_stages'] if stages is None else stages):
                stage = self.config['stages'][name]
                if stage['role'] != self.role or not stage.get('enabled', True):
                    continue
                record = states.get(name, {})
                forced = name in self.config.get('force_stages', []) and record.get('request') != self.run_id
                if self.current(name, task, states) and not forced:
                    if name == 'rectify' and str(folder) not in self.registered_rectifications:
                        if states[name].get('task') != task:
                            states[name]['task'] = dict(task)
                            write(folder / 'state.json', states)
                        update_rectified_index(self.config, self.values(task)['session'], True)
                        self.registered_rectifications.add(str(folder))
                    continue
                if not all(self.current(p, task, states) for p in stage.get('depends', [])):
                    continue
                signature = self.signature(name, task, states)
                if record.get('status') == 'failed' and record.get('request') == self.run_id:
                    continue  # Never loop/rebill a failed task within one invocation.
                if record.get('status') == 'failed' and record.get('signature') == signature and not (self.retry or forced):
                    continue
                values = self.values(task, gpu)
                if name in ('download', 'convert', 'qc', 'rectify') and 'rectify' in self.config['stages']:
                    self.registered_rectifications.discard(str(folder))
                    update_rectified_index(self.config, values['session'], False)
                self.invalidate(name, states)
                attempt = uuid4().hex
                history = folder / 'runs' / attempt
                history.mkdir(parents=True)
                running = dict(status='running', request=self.run_id, signature=signature,
                               generation=attempt, stage=name, role=self.role, started=time.time(),
                               host=os.environ.get('HOSTNAME', ''), pid=os.getpid(), log=str(history / 'console.log'))
                if name == 'rectify':
                    running['task'] = dict(task)
                states[name] = running
                write(folder / 'state.json', states)
                location = dict(role=self.role, worker=str(gpu))
                if self.role == 'gpu':
                    location['gpu'] = str(gpu)
                print(json.dumps(dict(batch=task['batch'], task=task['name'], stage=name,
                                      status='start', **location)), flush=True)
                backups = []
                try:
                    if forced:
                        for index, template in enumerate(stage.get('backup_paths', [])):
                            path = Path(template.format_map(values))
                            roots = [Path(values[k]).resolve() for k in ('input_root', 'output_root')]
                            if path.is_symlink() or not any(path.resolve().is_relative_to(r) and path.resolve() != r for r in roots):
                                raise ValueError('Unsafe backup target: ' + str(path))
                            if path.exists():
                                backup = history / 'previous' / str(index)
                                backup.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(path), str(backup))
                                backups.append((path, backup))
                                running['backups'] = [[str(p), str(b)] for p,b in backups]
                                write(folder / 'state.json', states)
                    # Adoption needs an explicit validator; never infer success from directory existence.
                    adopted = False
                    if self.adopt and not record and not forced and stage.get('adopt_command'):
                        check = self.command(stage['adopt_command'], values)
                        with (history / 'adopt.log').open('w', encoding='utf-8') as log:
                            adopted = subprocess.run(check, stdout=log, stderr=subprocess.STDOUT,
                                                     pass_fds=(acquired.fileno(),)).returncode == 0
                    command = self.command(stage['command'], values)
                    if name == 'qc' and getattr(self, 'calibration_donor_cache', None):
                        command.extend(['--calibration-donor-cache', self.calibration_donor_cache])
                    if forced:
                        command += self.command(stage['force_args'], values) if stage.get('force_args') else []
                    write(history / 'command.json', dict(command=command, adopted=adopted))
                    code = 0
                    if not adopted:
                        env = os.environ.copy()
                        env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                        if self.config.get('level') == 'L3':
                            env['ARGUS_L3_WORK_ROOT'] = str(Path(values['state_root']) / 'L3' / task['batch'] / task['name'] / 'review_work')
                            env['L3_RECTIFY_WORK_ROOT'] = str(Path(values['state_root']) / 'L3' / task['batch'] / task['name'] / 'rectify_work')
                            env['L3_DEPTH_WORK_ROOT'] = str(Path(values['state_root']) / 'L3' / task['batch'] / task['name'] / 'depth_work')
                        if getattr(self, 'gpu_executor', None) is not None:
                            code = self.gpu_executor(command, history / 'console.log')
                        else:
                            with (history / 'console.log').open('w', encoding='utf-8') as log:
                                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                           pass_fds=(acquired.fileno(),))
                                try:
                                    while True:
                                        try:
                                            code = process.wait(timeout=30)
                                            break
                                        except subprocess.TimeoutExpired:
                                            print(json.dumps(dict(batch=task['batch'],task=task['name'],
                                                stage=name,status='running',pid=process.pid,
                                                elapsed_s=round(time.time()-running['started']),log=str(history/'console.log'))),flush=True)
                                except BaseException:
                                    process.terminate()
                                    process.wait()
                                    raise
                    if code == 0 and stage.get('validate_command'):
                        with (history / 'validate.log').open('w', encoding='utf-8') as log:
                            code = subprocess.run(self.command(stage['validate_command'], values),
                                                  stdout=log, stderr=subprocess.STDOUT,
                                                  pass_fds=(acquired.fileno(),)).returncode
                    if code == 0 and self.artifacts(name, task) is None:
                        raise ValueError('Expected stage outputs are missing')
                    running.update(status='done' if code == 0 else 'failed', returncode=code, adopted=adopted)
                except Exception as exc:
                    running.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                if running['status'] == 'failed':
                    for index, (path, backup) in enumerate(reversed(backups)):
                        if path.exists():
                            retained = history / 'failed_outputs' / str(index)
                            retained.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(path), str(retained))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(backup), str(path))
                running['finished'] = time.time()
                # QC repairs/meta updates can change its inputs during this invocation.
                running['signature'] = self.signature(name, task, states)
                running['artifacts'] = self.artifacts(name, task)
                if name == 'rectify':
                    update_rectified_index(self.config, values['session'], running['status'] == 'done')
                    if running['status'] == 'done':
                        self.registered_rectifications.add(str(folder))
                write(history / 'result.json', running)
                if self.config.get('level') == 'L3' and running['status'] == 'done':
                    running['log'] = None
                write(folder / 'state.json', states)
                if self.config.get('level') == 'L3' and running['status'] == 'done':
                    cleanup_success_work(folder, name)
                    shutil.rmtree(history)
                print(json.dumps(dict(batch=task['batch'], task=task['name'], stage=name,
                                      status=running['status'], log=running['log'], **location)), flush=True)
                return True
            return False


class ResidentWorker:
    def __init__(self,config,stage,device):
        self.config,self.stage,self.device=config,stage,device
        self.process=None

    def execute(self,command,log):
        if self.process is None:
            python = (self.config['paths'].get('hand_python',command[0])
                      if self.stage=='hand' else command[0])
            entry=Path(__file__).with_name('resident.py')
            self.process=subprocess.Popen([str(python),'-u',str(entry),self.stage,
                self.config['level'],str(self.device)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,encoding='utf-8',bufsize=1)
        process=self.process
        try:
            process.stdin.write(json.dumps(dict(command=command,log=str(log)))+'\n')
            process.stdin.flush()
            line=process.stdout.readline()
            if not line:
                raise RuntimeError('Resident GPU worker exited: '+process.stderr.read()[-2000:])
            return json.loads(line)['returncode']
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is None:
            return
        process,self.process=self.process,None
        if process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def run_gpu_phases(runner, selected, devices, args):
    """Drain depth across the current JSON snapshot before starting hand tasks."""
    config = runner.config
    stages = [n for n in config['run_stages'] if config['stages'][n]['role'] == 'gpu'
              and config['stages'][n].get('enabled', True)]
    stages.sort(key=lambda n: (0 if n == 'depth' else 1 if n == 'hand' else 2))
    active_stage = None
    residents = []
    def close_workers():
        for worker in residents:
            if worker is not None:
                worker.close()
        residents.clear()
    def pending(stage, task):
        if not isinstance(runner, Runner):
            return True
        states = read(runner.folder(task) / 'state.json')
        record = states.get(stage, {})
        forced = stage in config.get('force_stages', []) and record.get('request') != runner.run_id
        if runner.current(stage, task, states) and not forced:
            return False
        if not all(runner.current(p, task, states) for p in config['stages'][stage].get('depends', [])):
            return False
        if record.get('status') == 'failed':
            if record.get('request') == runner.run_id:
                return False
            if record.get('signature') == runner.signature(stage, task, states) and not (runner.retry or forced):
                return False
        return True
    try:
        while True:
            for stage in stages:
                if not any(pending(stage, task) for task in selected):
                    continue
                slots = [device for device in devices
                         for _ in range(args.depth_workers_per_gpu if stage == 'depth' else 1)]
                if active_stage != stage:
                    close_workers()
                    residents.extend(ResidentWorker(config, stage, device)
                        if isinstance(runner, Runner) and stage in ('depth', 'hand') else None for device in slots)
                    active_stage = stage
                print(json.dumps(dict(phase=stage, gpu_slots=len(slots), sessions=len(selected))), flush=True)
                def lane(index):
                    import copy
                    local = copy.copy(runner)
                    worker = residents[index]
                    if worker is not None:
                        local.gpu_executor = worker.execute
                    while sum(local.step(task, slots[index], stages=[stage]) for task in selected):
                        pass
                with ThreadPoolExecutor(max_workers=len(slots)) as pool:
                    list(pool.map(lane, range(len(slots))))
            if not args.watch:
                incomplete = []
                for task in selected:
                    states = read(runner.folder(task) / 'state.json')
                    incomplete.extend((task['name'], n) for n in stages if not runner.current(n, task, states))
                print(json.dumps(dict(incomplete=incomplete)), flush=True)
                return int(bool(incomplete))
            print(json.dumps(dict(status='waiting', retained_model_stage=active_stage,
                                  index=str(Path(config['paths']['output_root']) / 'batch_*/rectification_done.json'))), flush=True)
            time.sleep(min(args.poll_s, 30))
            selected = gpu_tasks(config, args.batches)
    finally:
        close_workers()


def run_cpu_phases(runner, selected, devices, args):
    """Finish each CPU stage over a fixed selection before advancing."""
    if not selected:
        raise ValueError('No sessions selected')
    stages = [n for n in runner.config['run_stages']
              if runner.config['stages'][n]['role'] == 'cpu'
              and runner.config['stages'][n].get('enabled', True)]
    for name in stages:
        for parent in runner.config['stages'][name].get('depends', []):
            if parent in stages and stages.index(parent) >= stages.index(name):
                raise ValueError('CPU stages must follow dependency order: ' + parent + ' before ' + name)
    print(json.dumps(dict(status='selection', sessions=len(selected), phases=stages)), flush=True)
    selected=list(selected)
    excluded=[]
    continue_passed=runner.config.get('cpu_failure_policy','stop')=='continue_passed'

    def failed_dependency(name, task, states):
        for parent in runner.config['stages'][name].get('depends', []):
            if runner.current(parent, task, states):
                continue
            if states.get(parent, {}).get('status') == 'failed':
                return parent
            upstream = failed_dependency(parent, task, states)
            if upstream:
                return upstream
        return None

    for name in stages:
        print(json.dumps(dict(phase=name, status='started', sessions=len(selected))), flush=True)
        while True:
            def lane(device):
                return sum(runner.step(task, device, stages=[name]) for task in selected)
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                changed = sum(pool.map(lane, devices))
            incomplete = []
            failed = False
            passed_tasks=[]
            failed_tasks=[]
            failure_reasons=[]
            for task in selected:
                states = read(runner.folder(task) / 'state.json')
                record = states.get(name, {})
                forced_pending = (name in runner.config.get('force_stages', [])
                                  and record.get('request') != runner.run_id)
                if forced_pending or not runner.current(name, task, states):
                    incomplete.append(dict(batch=task['batch'], session=task['name'],
                                           status=record.get('status', 'pending')))
                    failed |= record.get('status') == 'failed'
                    cause = (name if record.get('status') == 'failed'
                             else failed_dependency(name, task, states))
                    if cause:
                        failed_tasks.append(task)
                        failure_reasons.append(dict(batch=task['batch'], session=task['name'],
                                                    failed_stage=cause, blocked_stage=name))
                else:
                    passed_tasks.append(task)
            if not incomplete:
                print(json.dumps(dict(phase=name, status='complete', sessions=len(selected))), flush=True)
                break
            if changed:
                continue
            print(json.dumps(dict(phase=name, status='blocked', incomplete=incomplete)), flush=True)
            if continue_passed and failed_tasks and len(failed_tasks)==len(incomplete):
                excluded.extend(failure_reasons)
                selected=passed_tasks
                print(json.dumps(dict(phase=name,status='filtered',passed=len(selected),
                                      excluded=len(failed_tasks))),flush=True)
                if not selected:
                    print(json.dumps(dict(status='finished',excluded=excluded,passed=0)),flush=True)
                    return 1
                break
            if (failed and not continue_passed) or not args.watch:
                return 1
            # Wait for externally locked work, without expanding the selected range.
            time.sleep(min(args.poll_s, 30))
    if continue_passed:
        print(json.dumps(dict(status='finished',passed=len(selected),excluded=excluded)),flush=True)
    return int(bool(excluded))


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--role', required=True, choices=('cpu', 'gpu'))
    parser.add_argument('--batches', nargs='*', default=[])
    parser.add_argument('--batch-range', help='CPU only: inclusive batch range, e.g. 1:12')
    parser.add_argument('--run-stages', help='Comma-separated override')
    parser.add_argument('--force-stages', help='Comma-separated override')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7', help='Physical GPU IDs')
    parser.add_argument('--depth-workers-per-gpu', type=int, default=None)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-s', type=float, default=5)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--adopt-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_range:
        if args.role != 'cpu' or args.batches:
            parser.error('--batch-range is CPU only and cannot be combined with --batches')
        try:
            first, last = map(int, args.batch_range.split(':'))
            if first < 1 or last < first:
                raise ValueError
        except ValueError:
            parser.error('--batch-range must be a positive inclusive range, e.g. 1:12')
        args.batches = [f'batch_{i:07d}' for i in range(first, last + 1)]
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.depth_workers_per_gpu is None:
        args.depth_workers_per_gpu = config.get('depth_workers_per_gpu', 2)
    for key in ('run_stages', 'force_stages'):
        value = getattr(args, key)
        if value is not None:
            config[key] = [v for v in value.split(',') if v]
    validate_config(config)
    if args.workers < 1 or args.depth_workers_per_gpu < 1 or args.poll_s <= 0:
        parser.error('workers and poll-s must be positive')
    runner = Runner(config, args.role, retry=args.retry_failed, adopt=args.adopt_existing)
    if args.role == 'cpu' and config.get('level') == 'L1' and 'qc' in config['run_stages']:
        command=config['stages']['qc']['command']
        if '--repair-calibration-root' in command and '--calibration-donor-cache' not in command:
            cache=Path(config['paths']['state_root'])/config['dataset']/'donor_snapshots'/(runner.run_id+'.json')
            runner.calibration_donor_cache=str(cache)
    selected = gpu_tasks(config, args.batches) if args.role == 'gpu' else tasks(config, args.batches)
    if args.dry_run:
        for task in selected:
            print(json.dumps(dict(task=task, stages=[n for n in config['run_stages'] if config['stages'][n]['role']==args.role])))
        return 0
    devices = args.gpus.split(',') if args.role == 'gpu' else [str(i) for i in range(args.workers)]
    if any(not d.strip() for d in devices) or len(set(devices)) != len(devices):
        parser.error('GPU IDs must be nonempty and unique')
    if args.role == 'gpu':
        return run_gpu_phases(runner, selected, devices, args)
    return run_cpu_phases(runner, selected, devices, args)


if __name__ == '__main__':
    raise SystemExit(main())
