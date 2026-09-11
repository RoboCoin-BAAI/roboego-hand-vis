from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest


def adapter():
    from roboego_hand_vis.pipelines.privacy import nodes
    assert hasattr(nodes, 'run_video_privacy_blur_node'), 'New privacy node must be exposed'
    return importlib.import_module('roboego_hand_vis.pipelines.privacy.video_blur')


def test_disabled_requires_no_runtime_or_paths(tmp_path, monkeypatch):
    mod = adapter()
    monkeypatch.chdir(tmp_path)
    assert mod.run_video_privacy_blur_node(None)['status'] == 'skipped'
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def configured(tmp_path, monkeypatch):
    mod = adapter()
    session = tmp_path / 'session with spaces'
    videos = session / 'videos'
    videos.mkdir(parents=True)
    for name in ['left', 'right', 'bright', 'left_blur', 'preview']:
        (videos / f'{name}.mp4').write_bytes(b'input')
    model = tmp_path / 'face weights.pt'
    model.write_bytes(b'weights')
    monkeypatch.setattr(mod.shutil, 'which', lambda value: sys.executable)
    return mod, {'enabled': True, 'python': sys.executable, 'session_dir': str(session), 'face_weights': str(model)}


def fake_success(args, *, env, cwd, log_path):
    output = Path(args[args.index('--output-dir') + 1])
    selected = [args[i + 1] for i, value in enumerate(args) if value == '--video']
    (output / '.batch').mkdir(exist_ok=True)
    for name in selected:
        (output / (Path(name).stem + '_blurred.mp4')).write_bytes(b'processed')
    (output / '.batch/summary.json').write_text(json.dumps({
        'counts': {'completed': len(selected), 'skipped': 0, 'failed': 0}, 'failures': [],
    }))
    return 0


def test_auto_only_selects_original_cameras_and_uses_project_package(configured, monkeypatch):
    mod, config = configured
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return fake_success(args, **kwargs)
    monkeypatch.setattr(mod, '_run_batch', run)
    report = mod.run_video_privacy_blur_node(config)
    assert report['status'] == 'complete'
    assert list(report['cameras']) == ['left', 'right', 'bright']
    assert report['counts'] == {'completed': 3, 'skipped': 0, 'failed': 0}
    args, kwargs = calls[0]
    assert args[:3] == [sys.executable, '-m', 'roboego_hand_vis.video_privacy_blur.batch']
    assert kwargs['env']['PYTHONPATH'].split(mod.os.pathsep)[0] == str(mod.PROJECT_ROOT)
    assert '--track-gap' in args and args[args.index('--track-gap') + 1] == '0'
    assert Path(report['output_dir'], 'session_report.json').is_file()
    assert not Path(config['session_dir'], 'videos/left_blurred.mp4').exists()


@pytest.mark.parametrize('selection', [['right'], 'right', 'right|left'])
def test_explicit_camera_subset(configured, monkeypatch, selection):
    mod, config = configured
    monkeypatch.setattr(mod, '_run_batch', fake_success)
    report = mod.run_video_privacy_blur_node({**config, 'camera_streams': selection})
    expected = selection if isinstance(selection, list) else selection.split('|')
    assert list(report['cameras']) == expected


@pytest.mark.parametrize('change', [
    {'session_dir': ''}, {'camera_streams': ['bleft']}, {'camera_streams': '../left'},
    {'camera_streams': []}, {'camera_streams': ['left', 'left']},
    {'conf': .1}, {'conf': float('nan')}, {'imgsz': 0}, {'imgsz': 1.5}, {'threads': 0},
    {'jobs': 0}, {'track_gap': -1}, {'track_scale': .5}, {'timeout': 0},
    {'retries': -1}, {'scale': .9}, {'blur_strength': 0}, {'device': 'invalid'}, {'method': 'invalid'},
    {'face_weights': '/nonexistent/weights.pt'}, {'enabled': 'false'},
])
def test_invalid_config_fails_without_output(configured, change):
    mod, config = configured
    with pytest.raises((ValueError, FileNotFoundError)):
        mod.run_video_privacy_blur_node({**config, **change})
    assert not Path(config['session_dir'], 'privacy').exists()


@pytest.mark.parametrize('suffix', ['', '/videos', '/videos/output'])
def test_output_cannot_overlap_input_directory(configured, suffix):
    mod, config = configured
    with pytest.raises(ValueError, match='overlap'):
        mod.run_video_privacy_blur_node({**config, 'output_dir': config['session_dir'] + suffix})


def test_no_original_cameras_fails(configured):
    mod, config = configured
    for name in ['left', 'right', 'bright']:
        Path(config['session_dir'], 'videos', name + '.mp4').unlink()
    with pytest.raises(ValueError, match='camera'):
        mod.run_video_privacy_blur_node(config)


@pytest.mark.parametrize('payload', [None, '{invalid', '{}', '{"counts":{"completed":99,"skipped":0,"failed":0},"failures":[]}'])
def test_missing_or_corrupt_batch_report_never_succeeds(configured, monkeypatch, payload):
    mod, config = configured
    def run(args, **kwargs):
        if payload is not None:
            output = Path(args[args.index('--output-dir') + 1])
            (output / '.batch').mkdir(exist_ok=True)
            (output / '.batch/summary.json').write_text(payload)
        return 0
    monkeypatch.setattr(mod, '_run_batch', run)
    with pytest.raises(RuntimeError, match='report'):
        mod.run_video_privacy_blur_node(config)
    report = json.loads(Path(config['session_dir'], 'privacy/video_privacy_blur/session_report.json').read_text())
    assert report['status'] == 'failed'


def test_stale_summary_is_not_reused(configured, monkeypatch):
    mod, config = configured
    monkeypatch.setattr(mod, '_run_batch', fake_success)
    mod.run_video_privacy_blur_node(config)
    monkeypatch.setattr(mod, '_run_batch', lambda *a, **kw: 0)
    with pytest.raises(RuntimeError, match='report'):
        mod.run_video_privacy_blur_node(config)


def test_partial_failure_is_reported_and_raised(configured, monkeypatch):
    mod, config = configured
    def run(args, **kwargs):
        fake_success(args, **kwargs)
        output = Path(args[args.index('--output-dir') + 1])
        (output / '.batch/summary.json').write_text(json.dumps({
            'counts': {'completed': 2, 'skipped': 0, 'failed': 1},
            'failures': [{'source': 'right.mp4', 'error': 'decode failed'}],
        }))
        return 1
    monkeypatch.setattr(mod, '_run_batch', run)
    with pytest.raises(RuntimeError, match='failed'):
        mod.run_video_privacy_blur_node(config)
    report = json.loads(Path(config['session_dir'], 'privacy/video_privacy_blur/session_report.json').read_text())
    assert report['counts']['failed'] == 1
    assert report['failures'][0]['error'] == 'decode failed'


def test_kedro_node_defaults_and_launcher_are_opt_in(tmp_path):
    import yaml
    from roboego_hand_vis.pipeline_registry import register_pipelines
    from roboego_hand_vis.pipeline_launcher import LauncherRequest, build_stage_invocations
    mod = adapter()
    pipeline = register_pipelines()['privacy']
    node = next(n for n in pipeline.nodes if n.name == 'video_privacy_blur')
    assert node.inputs == ['params:video_privacy_blur']
    assert node.outputs == ['video_privacy_blur_report']
    defaults = yaml.safe_load((mod.PROJECT_ROOT / 'conf/base/parameters.yml').read_text())['video_privacy_blur']
    assert defaults['enabled'] is False
    assert defaults['jobs'] == 1
    invocation = build_stage_invocations(LauncherRequest(dataset='raw', session=tmp_path, stage='video_privacy_blur'))[0]
    assert invocation.pipeline == 'privacy'
    assert invocation.node == 'video_privacy_blur'
    assert invocation.params == {'video_privacy_blur.enabled': True, 'video_privacy_blur.session_dir': str(tmp_path)}
    raw = yaml.safe_load((mod.PROJECT_ROOT / 'conf/datasets/raw.yml').read_text())
    assert 'video_privacy_blur' not in str(raw['workflows'])


def test_missing_interpreter_or_ffmpeg_fails_before_output(configured, monkeypatch):
    mod, config = configured
    with pytest.raises(FileNotFoundError, match='interpreter'):
        mod.run_video_privacy_blur_node({**config, 'python': '/missing/python'})
    monkeypatch.setattr(mod.shutil, 'which', lambda value: None)
    with pytest.raises(FileNotFoundError, match='ffmpeg'):
        mod.run_video_privacy_blur_node(config)
    assert not Path(config['session_dir'], 'privacy').exists()


def test_camera_symlink_is_rejected(configured):
    mod, config = configured
    source = Path(config['session_dir'], 'videos/left.mp4')
    source.unlink()
    source.symlink_to(source.with_name('right.mp4'))
    with pytest.raises(FileNotFoundError, match='symlink'):
        mod.run_video_privacy_blur_node(config)


def test_nonzero_exit_without_summary_is_failure(configured, monkeypatch):
    mod, config = configured
    monkeypatch.setattr(mod, '_run_batch', lambda *a, **kw: 2)
    with pytest.raises(RuntimeError, match='exit 2'):
        mod.run_video_privacy_blur_node(config)
    report = json.loads(Path(config['session_dir'], 'privacy/video_privacy_blur/session_report.json').read_text())
    assert report['status'] == 'failed'
    assert report['returncode'] == 2


def test_missing_output_is_failure(configured, monkeypatch):
    mod, config = configured
    def run(args, **kwargs):
        code = fake_success(args, **kwargs)
        Path(args[args.index('--output-dir') + 1], 'left_blurred.mp4').unlink()
        return code
    monkeypatch.setattr(mod, '_run_batch', run)
    with pytest.raises(RuntimeError, match='missing output'):
        mod.run_video_privacy_blur_node(config)


def test_interruption_records_status(configured, monkeypatch):
    mod, config = configured
    def run(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(mod, '_run_batch', run)
    with pytest.raises(KeyboardInterrupt):
        mod.run_video_privacy_blur_node(config)
    report = json.loads(Path(config['session_dir'], 'privacy/video_privacy_blur/session_report.json').read_text())
    assert report['status'] == 'interrupted'


def test_session_lock_refuses_second_run(configured, monkeypatch):
    mod, config = configured
    output = Path(config['session_dir'], 'privacy/video_privacy_blur')
    with mod._session_lock(output):
        with pytest.raises(RuntimeError, match='another session'):
            mod.run_video_privacy_blur_node(config)


def test_imports_and_registration_do_not_load_inference_libraries():
    import subprocess
    mod = adapter()
    script = '''
import sys
blocked = {'ultralytics', 'torch', 'cv2'}
class BlockInference:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked or fullname.startswith('roboego_hand_vis.video_privacy_blur'):
            raise AssertionError('Inference import in Kedro: ' + fullname)
sys.meta_path.insert(0, BlockInference())
from roboego_hand_vis.pipelines.privacy.nodes import run_video_privacy_blur_node
assert run_video_privacy_blur_node(None)['status'] == 'skipped'
blocked = {'ultralytics'}
from roboego_hand_vis.pipeline_registry import register_pipelines
assert any(n.name == 'video_privacy_blur' for n in register_pipelines()['privacy'].nodes)
'''
    subprocess.run([sys.executable, '-c', script], cwd=mod.PROJECT_ROOT, check=True)


def test_batch_summary_cannot_override_adapter_metadata(configured, monkeypatch):
    mod, config = configured
    def run(args, **kwargs):
        code = fake_success(args, **kwargs)
        summary = Path(args[args.index('--output-dir') + 1], '.batch/summary.json')
        payload = json.loads(summary.read_text())
        payload.update(cameras={}, stage='untrusted', output_dir='/wrong')
        summary.write_text(json.dumps(payload))
        return code
    monkeypatch.setattr(mod, '_run_batch', run)
    report = mod.run_video_privacy_blur_node(config)
    assert report['stage'] == 'video_privacy_blur'
    assert len(report['cameras']) == 3


def test_interrupt_stops_supervisor_group_and_waits(tmp_path, monkeypatch):
    mod = adapter()
    calls = []
    class Process:
        pid = 123456
        waits = 0
        def wait(self):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt()
            calls.append('reaped')
            return -15
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda *a, **kw: Process())
    monkeypatch.setattr(mod.os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        mod._run_batch(['python'], env={}, cwd=tmp_path, log_path=tmp_path / 'run.log')
    assert calls == [(123456, mod.signal.SIGTERM), 'reaped']


def test_launcher_dry_run_creates_no_state_or_process(tmp_path, monkeypatch, capsys):
    from roboego_hand_vis.pipeline_launcher import LauncherRequest, run_launcher_request
    mod = adapter()
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run must not launch a process')
    monkeypatch.setattr(mod.subprocess, 'run', forbidden)
    commands = run_launcher_request(LauncherRequest(dataset='raw', session=tmp_path, stage='video_privacy_blur', dry_run=True))
    assert '--nodes video_privacy_blur' in commands[0]
    assert commands[0] in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_sigterm_during_calibration_stops_actual_child(tmp_path):
    import os
    import signal
    import subprocess
    import time
    mod = adapter()
    ready, stopped = tmp_path / 'ready', tmp_path / 'stopped'
    calibration = tmp_path / 'calibration.py'
    calibration.write_text('''
import signal, sys, time
from pathlib import Path
def stop(signum, frame):
    Path(sys.argv[2]).write_text('stopped')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).write_text('ready')
time.sleep(30)
''')
    supervisor = tmp_path / 'supervisor.py'
    supervisor.write_text('''
import subprocess, sys
# Deliberately no SIGTERM handler, matching early batch resource planning.
subprocess.run([sys.executable, *sys.argv[1:]], check=True)
''')
    driver = '''
import os, sys
from pathlib import Path
from roboego_hand_vis.pipelines.privacy.video_blur import _run_batch
try:
    _run_batch([sys.executable, *sys.argv[1:]], env=os.environ.copy(), cwd=Path.cwd(), log_path=Path(sys.argv[-1]).with_suffix('.log'))
except KeyboardInterrupt:
    sys.exit(130)
'''
    process = subprocess.Popen([sys.executable, '-c', driver, str(supervisor), str(calibration), str(ready), str(stopped)], cwd=mod.PROJECT_ROOT)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.02)
        assert ready.exists(), 'Calibration child did not start'
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 130
        deadline = time.monotonic() + 5
        while not stopped.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert stopped.read_text() == 'stopped'
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def test_privacy_backend_is_part_of_main_package():
    import importlib
    mod = adapter()
    package = mod.PROJECT_ROOT / 'roboego_hand_vis/video_privacy_blur'
    assert (package / '__init__.py').is_file()
    assert not (mod.PROJECT_ROOT / 'third_party/video_privacy_blur').exists()
    backend = importlib.import_module('roboego_hand_vis.video_privacy_blur.batch')
    assert Path(backend.__file__).parent == package
    assert not (package / 'pyproject.toml').exists()


def test_default_privacy_waits_for_all_processing_stages_but_named_pipeline_is_independent():
    from roboego_hand_vis.pipeline_registry import register_pipelines
    pipelines = register_pipelines()
    upstream_names = ('quality', 'semantic', 'datasets', 'restoration', 'depth', 'egohand', 'objects')
    expected = set().union(*(pipelines[name].outputs() for name in upstream_names))
    default = pipelines['__default__']
    privacy_node = next(node for node in default.nodes if node.name == 'video_privacy_blur')
    assert set(privacy_node.inputs) == {'params:video_privacy_blur', *expected}
    positions = {node.name: i for i, group in enumerate(default.grouped_nodes) for node in group}
    for stage in upstream_names:
        for node in pipelines[stage].nodes:
            assert positions[node.name] < positions['video_privacy_blur']
    standalone = next(node for node in pipelines['privacy'].nodes if node.name == 'video_privacy_blur')
    assert standalone.inputs == ['params:video_privacy_blur']


def test_completion_dependencies_execute_before_privacy_with_thread_runner(monkeypatch):
    from kedro.io import DataCatalog, MemoryDataset
    from kedro.pipeline import Pipeline, node
    from kedro.runner import ThreadRunner
    from roboego_hand_vis.pipelines.privacy import pipeline as privacy_pipeline
    completed = set()

    def depth():
        completed.add('depth')
        return {'status': 'complete'}

    def objects():
        completed.add('objects')
        return {'status': 'complete'}

    def blur(config):
        assert completed == {'depth', 'objects'}
        assert config == {'enabled': True}
        return {'status': 'complete'}

    monkeypatch.setattr(privacy_pipeline, 'run_video_privacy_blur_node', blur)
    graph = Pipeline([node(depth, None, 'depth_done'), node(objects, None, 'objects_done')])
    graph += privacy_pipeline.create_pipeline(after=['depth_done', 'objects_done'])
    catalog = DataCatalog({
        'params:video_privacy_blur': MemoryDataset(data={'enabled': True}),
        'params:egoblur_temporal_face_blur': MemoryDataset(data={'enabled': False}),
    })
    result = ThreadRunner(max_workers=3).run(graph, catalog)
    assert result['video_privacy_blur_report'].load()['status'] == 'complete'


def test_default_weights_are_in_project_and_match_source_record():
    import hashlib
    import yaml
    mod = adapter()
    parameters = yaml.safe_load((mod.PROJECT_ROOT / 'conf/base/parameters.yml').read_text())
    assert parameters['video_privacy_blur']['face_weights'] == 'conf/models/yolov8n-face.pt'
    assert mod.DEFAULTS['face_weights'] == 'conf/models/yolov8n-face.pt'
    weights = mod.PROJECT_ROOT / 'conf/models/yolov8n-face.pt'
    metadata = json.loads(weights.with_suffix('.pt.source.json').read_text())
    assert weights.stat().st_size == metadata['bytes']
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == metadata['sha256']
