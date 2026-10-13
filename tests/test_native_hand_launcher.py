import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import shutil
import os
import signal
import time
from unittest.mock import MagicMock, patch

import pytest
import yaml
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location(
    'native_hand', Path(__file__).resolve().parents[1] / 'roboego_hand_vis/hand/run.py')
hand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hand)


def write_video(path, count=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (32, 32))
    assert writer.isOpened()
    for _ in range(count):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()


def write_prediction(path, count=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pose_enc=np.ones((count, 9)), hand=np.ones((count, 218)),
             hand_presence_logits=np.ones((count, 2)))


def write_final(final, count=3):
    final.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({'frame_idx': list(range(count))}), final / 'hand_tracking.parquet')
    write_video(final / 'render.mp4', count)
    (final / 'summary.json').write_text('{"fixture": true}')
    (final / 'results.pkl').write_bytes(b'intermediate')


def install_stages(monkeypatch, failure=None):
    calls = []
    real_run = hand.run_stage
    def run(stage, environment, log, **kwargs):
        name = stage['name']
        if name.startswith('validate_'):
            return real_run(stage, environment, log, **kwargs)
        calls.append(name)
        with log.open('a') as stream:
            stream.write(f'fixture {name}\n')
        command = stage['command']
        if name == 'mint':
            if failure == 'mint':
                return 7
            if failure != 'missing':
                write_prediction(Path(command[command.index('--output') + 1]) / 'prediction.npz')
        else:
            if failure == 'interrupt':
                raise KeyboardInterrupt
            write_final(Path(command[-1]) / 'mint/fixture/final')
            if failure == 'ego':
                return 9
        return 0
    monkeypatch.setattr(hand, 'run_stage', run)
    return calls


def log_records(output):
    return [json.loads(line[len('[launcher] '):]) for line in (output / 'run.log').read_text().splitlines()
            if line.startswith('[launcher] ')]


@pytest.fixture
def plan(tmp_path):
    mint, ego, source = [tmp_path / name for name in ('mint repo', 'ego repo', 'input session')]
    for folder in (mint / 'mint', mint / 'scripts', ego / 'scripts', source / 'videos', source / 'depth'):
        folder.mkdir(parents=True)
    (mint / 'model.safetensors').write_bytes(b'weights')
    (ego / 'run.py').touch()
    (ego / 'scripts/run_without_front.sh').write_text('# EGOHANDKIT_PYTHON')
    (ego / 'scripts/export_egohandkit_cache_mint.py').write_text('exporter')
    write_video(source / 'videos/left_rectified.mp4')
    config = {'paths': {'mint_root': str(mint), 'egohandkit_root': str(ego),
                        'mint_python': sys.executable, 'egohandkit_python': sys.executable,
                        'mint_checkpoint': 'model.safetensors'}}
    return hand.build_plan(config, source, tmp_path / 'new output')


def test_original_command_contract(plan):
    command = plan['stages'][0]['command']
    assert command[1:4] == ['-m', 'mint', 'infer']
    assert command[command.index('--camera-mode'):] == [
        '--camera-mode', 'chunked', '--max-frames', '10000', '--target-fps', '30',
        '--window', '32', '--window-batch-size', '1', '--no-render']
    assert plan['stages'][1]['command'][2:] == [
        'mint', plan['video'], plan['prediction'], plan['depth'],
        str(Path(plan['output_dir']) / '.work/attempts/egohandkit-0001')]
    hand.preflight(plan)
    assert not Path(plan['output_dir']).exists()


def test_python_symlink_is_not_resolved_out_of_venv(tmp_path):
    python = tmp_path / 'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    config = {'paths': {'mint_root': str(tmp_path), 'egohandkit_root': str(tmp_path),
                        'mint_python': str(python), 'egohandkit_python': str(python),
                        'mint_checkpoint': 'model.safetensors'}}
    plan = hand.build_plan(config, tmp_path, tmp_path / 'output')
    assert plan['stages'][0]['command'][0] == str(python)
    assert plan['environment']['EGOHANDKIT_PYTHON'] == str(python)


@pytest.mark.parametrize('failure,expected_calls,code', [(None, 2, 0), ('mint', 1, 7),
                                                        ('missing', 1, 1), ('ego', 2, 9)])
def test_pipeline_success_and_stop_on_failure(plan, monkeypatch, failure, expected_calls, code):
    calls = install_stages(monkeypatch, failure)
    assert hand.execute(plan) == code
    assert len(calls) == expected_calls
    output = Path(plan['output_dir'])
    records = log_records(output)
    assert records[-1]['returncode'] == code
    assert any(record['event'] == ('complete' if code == 0 else 'failed') for record in records)
    if code == 0:
        assert {path.name for path in output.iterdir()} == {'hand_tracking.parquet', 'render.mp4', 'run.log'}
    else:
        assert (output / '.work/state.json').is_file()
    with pytest.raises(FileExistsError):
        hand.execute(plan)


def test_conflicting_exporter_is_not_overwritten(plan):
    exporter = Path(plan['environment']['MINT_ROOT']) / 'scripts/export_egohandkit_cache.py'
    exporter.write_text('user edits')
    with pytest.raises(ValueError, match='exporter differs'):
        hand.preflight(plan)
    assert exporter.read_text() == 'user edits'


def test_dry_run_needs_no_environment_and_creates_no_output(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'paths': {
        'mint_root': '/unavailable/mint', 'egohandkit_root': '/unavailable/ego',
        'mint_python': '/unavailable/python', 'egohandkit_python': '/unavailable/python',
        'mint_checkpoint': 'checkpoints/model.safetensors'}}))
    output = tmp_path / 'output'
    assert hand.main(['--config', str(config), '--input_dir', '/unavailable/input',
                      '--output_dir', str(output), '--dry-run']) == 0
    assert not output.exists()


def test_subprocess_cwd_environment_and_log(tmp_path):
    log = tmp_path / 'stage.log'
    stage = {'cwd': str(tmp_path), 'command': [sys.executable, '-c',
             'import os; print(os.getcwd()); print(os.environ["TEST_NATIVE"]); raise SystemExit(3)']}
    assert hand.run_stage(stage, {'TEST_NATIVE': 'passed'}, log) == 3
    assert log.read_text().splitlines() == [str(tmp_path), 'passed']


def test_interrupt_stops_child_process_group(tmp_path):
    process = MagicMock(pid=12345)
    from itertools import chain, repeat
    process.wait.side_effect = chain([KeyboardInterrupt], repeat(0))
    process.poll.return_value = None
    with patch.object(hand.subprocess, 'Popen', return_value=process), \
            patch.object(hand.os, 'killpg') as kill, \
            patch('roboego_hand_vis.common.processes.group_has_live_processes', return_value=False):
        with pytest.raises(KeyboardInterrupt):
            hand.run_stage({'command': ['fake'], 'cwd': str(tmp_path)}, {}, tmp_path / 'log')
    kill.assert_any_call(12345, hand.signal.SIGTERM)
    assert process.wait.call_count >= 2


def test_same_checkout_lock_refuses_second_launch(plan):
    key = hand.hashlib.sha256(plan['environment']['EGOHANDKIT_ROOT'].encode()).hexdigest()[:24]
    path = Path(hand.tempfile.gettempdir()) / f'roboego-native-hand-{hand.os.getuid()}-{key}.lock'
    with path.open('a') as lock:
        hand.fcntl.flock(lock, hand.fcntl.LOCK_EX | hand.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another native hand launch'):
            hand.execute(plan)
    assert not Path(plan['output_dir']).exists()


@pytest.mark.parametrize('mint_exit,full', [(0, False), (0, True), (7, False)])
def test_real_module_cli_and_subprocess_chain(plan, tmp_path, mint_exit, full):
    mint = Path(plan['environment']['MINT_ROOT'])
    ego = Path(plan['environment']['EGOHANDKIT_ROOT'])
    (mint / 'mint/__init__.py').touch()
    (mint / 'mint/__main__.py').write_text(
        'import sys\nfrom pathlib import Path\nimport numpy as np\n'
        f'if {mint_exit}: raise SystemExit({mint_exit})\n'
        'out=Path(sys.argv[sys.argv.index("--output")+1]); out.mkdir(parents=True)\n'
        'np.savez(out/"prediction.npz", pose_enc=np.ones((3,9)), hand=np.ones((3,218)), hand_presence_logits=np.ones((3,2)))\n'
        'print("fake MINT finished")\n')
    write_final(ego / 'fixture')
    (ego / 'scripts/run_without_front.sh').write_text(
        '#!/bin/bash\nset -eu\n# EGOHANDKIT_PYTHON\n'
        'test -s "$3"\ntest "$1" = mint\n'
        'mkdir -p "$5/mint/fixture/final"\n'
        'cp "$EGOHANDKIT_ROOT"/fixture/* "$5/mint/fixture/final/"\n'
        'printf "%s\\n" "$EGOHANDKIT_ROOT"\n')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'paths': {
        'mint_root': str(mint), 'egohandkit_root': str(ego),
        'mint_python': sys.executable, 'egohandkit_python': sys.executable,
        'mint_checkpoint': plan['checkpoint']}}))
    result = subprocess.run([
        sys.executable, '-m', 'roboego_hand_vis.hand.run', '--config', str(config),
        '--input_dir', plan['input_dir'], '--output_dir', plan['output_dir'],
        *(['--full-output'] if full else [])],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
    assert result.returncode == mint_exit, result.stderr
    output = Path(plan['output_dir'])
    assert log_records(output)[-1]['returncode'] == mint_exit
    if mint_exit == 0:
        assert str(ego) in (output / 'run.log').read_text()
        assert (output / '.work').exists() == full
        assert pq.read_metadata(output / 'hand_tracking.parquet').num_rows == 3


@pytest.mark.parametrize('failure,expected', [('mint', ['mint', 'egohandkit']),
                                            ('ego', ['egohandkit']), ('interrupt', ['egohandkit'])])
def test_resume_reuses_only_completed_stages(plan, monkeypatch, failure, expected):
    install_stages(monkeypatch, failure)
    assert hand.execute(plan) != 0
    output = Path(plan['output_dir'])
    old_log = (output / 'run.log').read_text()
    calls = install_stages(monkeypatch)
    assert hand.execute(plan, resume=True) == 0
    assert calls == expected
    assert (output / 'run.log').read_text().startswith(old_log)
    assert not (output / '.work').exists()
    assert any(record.get('attempt') == 2 for record in log_records(output))


def test_full_output_retains_attempts_and_resumes_completed_delivery(plan, monkeypatch):
    plan['output_mode'] = 'full'
    install_stages(monkeypatch, 'ego')
    assert hand.execute(plan) == 9
    calls = install_stages(monkeypatch)
    assert hand.execute(plan, resume=True) == 0
    output = Path(plan['output_dir'])
    attempts = output / '.work/attempts'
    assert {path.name for path in attempts.iterdir()} == {'mint-0001', 'egohandkit-0001', 'egohandkit-0002'}
    assert hand.execute(plan, resume=True) == 0
    assert calls == ['egohandkit']


@pytest.mark.parametrize('change', ['input', 'model', 'depth', 'source', 'parameters', 'assets'])
def test_resume_refuses_changed_identity(plan, monkeypatch, change):
    install_stages(monkeypatch, 'ego')
    assert hand.execute(plan) == 9
    if change == 'parameters':
        plan['stages'][0]['command'][-2] = '2'
    else:
        path = {'input': Path(plan['video']), 'model': Path(plan['checkpoint']),
                'depth': Path(plan['depth']) / 'new.npy',
                'assets': Path(plan['environment']['EGOHANDKIT_ROOT']) / '_DATA/weights.ckpt',
                'source': Path(plan['environment']['MINT_ROOT']) / 'mint/model.py'}[change]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'changed')
    calls = install_stages(monkeypatch)
    with pytest.raises(ValueError, match='differs|changed'):
        hand.execute(plan, resume=True)
    assert not calls


def test_auto_copied_exporter_does_not_invalidate_resume(plan, monkeypatch):
    install_stages(monkeypatch, 'ego')
    assert hand.execute(plan) == 9
    mint = Path(plan['environment']['MINT_ROOT'])
    ego = Path(plan['environment']['EGOHANDKIT_ROOT'])
    shutil.copyfile(ego / 'scripts/export_egohandkit_cache_mint.py', mint / 'scripts/export_egohandkit_cache.py')
    calls = install_stages(monkeypatch)
    assert hand.execute(plan, resume=True) == 0
    assert calls == ['egohandkit']


def test_resume_refuses_corrupt_completed_prediction(plan, monkeypatch):
    install_stages(monkeypatch, 'ego')
    assert hand.execute(plan) == 9
    Path(plan['prediction']).write_bytes(b'corrupt')
    calls = install_stages(monkeypatch)
    assert hand.execute(plan, resume=True) == 1
    assert not calls
    assert Path(plan['prediction']).exists()


@pytest.mark.parametrize('point', ['before_publish', 'after_publish', 'cleanup'])
def test_resume_publication_and_cleanup_without_inference(plan, monkeypatch, point):
    calls = install_stages(monkeypatch)
    publish, cleanup = hand.outputs.publish, hand.outputs.cleanup_work
    def interrupted_publish(final, output):
        if point == 'after_publish':
            publish(final, output)
        raise KeyboardInterrupt
    def interrupted_cleanup(output):
        shutil.rmtree(output / '.work/attempts')
        raise KeyboardInterrupt
    if point == 'cleanup':
        monkeypatch.setattr(hand.outputs, 'cleanup_work', interrupted_cleanup)
    else:
        monkeypatch.setattr(hand.outputs, 'publish', interrupted_publish)
    assert hand.execute(plan) == 130
    assert calls == ['mint', 'egohandkit']
    monkeypatch.setattr(hand.outputs, 'publish', publish)
    monkeypatch.setattr(hand.outputs, 'cleanup_work', cleanup)
    assert hand.execute(plan, resume=True) == 0
    assert calls == ['mint', 'egohandkit']
    assert not (Path(plan['output_dir']) / '.work').exists()


def test_cleanup_checks_delivery_before_deleting(plan, monkeypatch):
    install_stages(monkeypatch)
    cleanup = hand.outputs.cleanup_work
    monkeypatch.setattr(hand.outputs, 'cleanup_work', lambda output: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert hand.execute(plan) == 130
    output = Path(plan['output_dir'])
    (output / 'render.mp4').write_bytes(b'corrupt')
    monkeypatch.setattr(hand.outputs, 'cleanup_work', cleanup)
    assert hand.execute(plan, resume=True) == 1
    assert (output / '.work/attempts').is_dir()


@pytest.mark.parametrize('target', ['.work', 'run.log'])
def test_resume_rejects_symlinked_control_paths(plan, tmp_path, target):
    output = Path(plan['output_dir'])
    output.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep').write_text('untouched')
    (output / target).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        hand.execute(plan, resume=True)
    assert (outside / 'keep').read_text() == 'untouched'


def test_compacted_result_cannot_be_resumed_or_overwritten(plan, monkeypatch):
    install_stages(monkeypatch)
    assert hand.execute(plan) == 0
    with pytest.raises(ValueError, match='already compacted'):
        hand.execute(plan, resume=True)
    with pytest.raises(FileExistsError):
        hand.execute(plan)


@pytest.mark.parametrize('damage', ['missing', 'empty', 'corrupt', 'mismatch', 'multiple', 'symlink'])
def test_invalid_final_output_never_cleans_intermediates(plan, monkeypatch, tmp_path, damage):
    install_stages(monkeypatch)
    runner = hand.run_stage
    def damaged(stage, environment, log, **kwargs):
        code = runner(stage, environment, log, **kwargs)
        if stage['name'] == 'egohandkit':
            final = Path(stage['command'][-1]) / 'mint/fixture/final'
            parquet = final / 'hand_tracking.parquet'
            if damage == 'missing':
                (final / 'render.mp4').unlink()
            elif damage in ('empty', 'corrupt'):
                parquet.write_bytes(b'' if damage == 'empty' else b'not parquet')
            elif damage == 'mismatch':
                write_video(final / 'render.mp4', 2)
            elif damage == 'multiple':
                write_final(final.parent.parent / 'other/final')
            else:
                outside = tmp_path / 'outside.parquet'
                shutil.copyfile(parquet, outside)
                parquet.unlink()
                parquet.symlink_to(outside)
        return code
    monkeypatch.setattr(hand, 'run_stage', damaged)
    assert hand.execute(plan) == 1
    output = Path(plan['output_dir'])
    assert Path(plan['prediction']).is_file()
    assert (output / '.work/state.json').exists()
    assert not (output / 'hand_tracking.parquet').exists()


def test_output_lock_is_independent_of_checkout(plan):
    key = hand.hashlib.sha256(plan['output_dir'].encode()).hexdigest()[:24]
    path = Path(hand.tempfile.gettempdir()) / f'roboego-hand-output-{hand.os.getuid()}-{key}.lock'
    with path.open('a') as lock:
        hand.fcntl.flock(lock, hand.fcntl.LOCK_EX | hand.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='checkout or output'):
            hand.execute(plan)
    assert not Path(plan['output_dir']).exists()


def test_checkpoint_atomic_replace_failure_preserves_previous_state(tmp_path, monkeypatch):
    from roboego_hand_vis.hand import state

    checkpoint = tmp_path / 'state.json'
    state.atomic_json(checkpoint, {'version': 'previous'})
    def fail_replace(*args):
        raise OSError('simulated disk error')
    monkeypatch.setattr(state.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='disk error'):
        state.atomic_json(checkpoint, {'version': 'next'})
    assert json.loads(checkpoint.read_text()) == {'version': 'previous'}
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_depth_identity_follows_directory_symlink_and_rejects_cycles(tmp_path):
    from roboego_hand_vis.hand.state import tree_identity

    depth, external = tmp_path / 'depth', tmp_path / 'external'
    depth.mkdir()
    external.mkdir()
    (external / 'frame.png').write_bytes(b'first')
    (depth / 'frames').symlink_to(external, target_is_directory=True)
    before = tree_identity(depth)
    (external / 'frame.png').write_bytes(b'changed')
    assert tree_identity(depth) != before
    (external / 'cycle').symlink_to(depth, target_is_directory=True)
    with pytest.raises(ValueError, match='cycle'):
        tree_identity(depth)


def test_cleanup_never_follows_nested_external_symlinks(tmp_path):
    output, external = tmp_path / 'output', tmp_path / 'external'
    (output / '.work/attempts').mkdir(parents=True)
    external.mkdir()
    (external / 'keep').write_text('original')
    (output / '.work/state.json').write_text('{}')
    (output / '.work/attempts/linked-cache').symlink_to(external, target_is_directory=True)
    hand.outputs.cleanup_work(output)
    assert (external / 'keep').read_text() == 'original'
    assert not (output / '.work').exists()


def test_inherited_lock_survives_launcher_sigkill(tmp_path):
    lock_path, marker = tmp_path / 'lock', tmp_path / 'child.pid'
    child_code = f'import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)'
    script = (
        'import fcntl\nfrom pathlib import Path\nfrom roboego_hand_vis.hand.run import run_stage\n'
        f'with Path({str(lock_path)!r}).open("a") as lock:\n'
        ' fcntl.flock(lock, fcntl.LOCK_EX)\n'
        f' run_stage({{"cwd": {str(tmp_path)!r}, "command": [{sys.executable!r}, "-c", {child_code!r}]}}, '
        f'{{}}, Path({str(tmp_path / "run.log")!r}), lock_fds=(lock.fileno(),))\n')
    parent = subprocess.Popen([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1])
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        parent.kill()
        parent.wait(timeout=5)
        with lock_path.open('a') as lock:
            with pytest.raises(BlockingIOError):
                hand.fcntl.flock(lock, hand.fcntl.LOCK_EX | hand.fcntl.LOCK_NB)
    finally:
        if marker.exists():
            try:
                os.killpg(int(marker.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)


def test_dry_run_modes_and_direct_script_entry(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'paths': {
        'mint_root': '/missing/mint', 'egohandkit_root': '/missing/ego',
        'mint_python': '/missing/python', 'egohandkit_python': '/missing/python',
        'mint_checkpoint': 'model'}}))
    result = subprocess.run([sys.executable, str(Path(hand.__file__).resolve()),
                             '--config', str(config), '--input_dir', '/missing/input',
                             '--output_dir', str(tmp_path / 'output'), '--dry-run', '--resume', '--full-output'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"output_mode": "full"' in result.stdout
    assert not (tmp_path / 'output').exists()
