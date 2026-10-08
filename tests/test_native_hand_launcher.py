import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    'native_hand', Path(__file__).resolve().parents[1] / 'roboego_hand_vis/hand/run.py')
hand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hand)


@pytest.fixture
def plan(tmp_path):
    mint, ego, source = [tmp_path / name for name in ('mint repo', 'ego repo', 'input session')]
    for folder in (mint / 'mint', mint / 'scripts', ego / 'scripts', source / 'videos', source / 'depth'):
        folder.mkdir(parents=True)
    (mint / 'model.safetensors').write_bytes(b'weights')
    (ego / 'run.py').touch()
    (ego / 'scripts/run_without_front.sh').write_text('# EGOHANDKIT_PYTHON')
    (ego / 'scripts/export_egohandkit_cache_mint.py').write_text('exporter')
    (source / 'videos/left_rectified.mp4').write_bytes(b'video')
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
        'mint', plan['video'], plan['prediction'], plan['depth'], plan['output_dir']]
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
    calls = []
    def stage(stage, environment, log):
        calls.append(stage['name'])
        log.write_text('fixture log')
        if stage['name'] == 'mint':
            if failure == 'mint':
                return 7
            if failure != 'missing':
                prediction = Path(plan['prediction'])
                prediction.parent.mkdir()
                prediction.write_bytes(b'npz')
        return 9 if stage['name'] == 'egohandkit' and failure == 'ego' else 0
    monkeypatch.setattr(hand, 'run_stage', stage)
    assert hand.execute(plan) == code
    assert len(calls) == expected_calls
    report = json.loads((Path(plan['output_dir']) / 'run_report.json').read_text())
    assert report['returncode'] == code
    assert report['quality_status'] == 'not_validated'
    assert report['status'] == ('complete' if code == 0 else 'failed')
    assert (Path(plan['output_dir']) / 'mint.log').exists()
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
    process.wait.side_effect = [KeyboardInterrupt, 0]
    process.poll.return_value = None
    with patch.object(hand.subprocess, 'Popen', return_value=process), \
            patch.object(hand.os, 'killpg') as kill:
        with pytest.raises(KeyboardInterrupt):
            hand.run_stage({'command': ['fake'], 'cwd': str(tmp_path)}, {}, tmp_path / 'log')
    kill.assert_called_once_with(12345, hand.signal.SIGTERM)
    assert process.wait.call_count == 2


def test_same_checkout_lock_refuses_second_launch(plan):
    key = hand.hashlib.sha256(plan['environment']['EGOHANDKIT_ROOT'].encode()).hexdigest()[:24]
    path = Path(hand.tempfile.gettempdir()) / f'roboego-native-hand-{hand.os.getuid()}-{key}.lock'
    with path.open('a') as lock:
        hand.fcntl.flock(lock, hand.fcntl.LOCK_EX | hand.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another native hand launch'):
            hand.execute(plan)
    assert not Path(plan['output_dir']).exists()


@pytest.mark.parametrize('mint_exit', [0, 7])
def test_real_module_cli_and_subprocess_chain(plan, tmp_path, mint_exit):
    mint = Path(plan['environment']['MINT_ROOT'])
    ego = Path(plan['environment']['EGOHANDKIT_ROOT'])
    (mint / 'mint/__init__.py').touch()
    (mint / 'mint/__main__.py').write_text(
        'import sys\nfrom pathlib import Path\n'
        f'if {mint_exit}: raise SystemExit({mint_exit})\n'
        'out=Path(sys.argv[sys.argv.index("--output")+1]); out.mkdir()\n'
        '(out/"prediction.npz").write_bytes(b"fixture only")\n'
        'print("fake MINT finished")\n')
    (ego / 'scripts/run_without_front.sh').write_text(
        '#!/bin/bash\nset -eu\n# EGOHANDKIT_PYTHON\n'
        'test -s "$3"\ntest "$1" = mint\n'
        'printf "%s\\n" "$EGOHANDKIT_ROOT" > "$5/ego_called"\n')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'paths': {
        'mint_root': str(mint), 'egohandkit_root': str(ego),
        'mint_python': sys.executable, 'egohandkit_python': sys.executable,
        'mint_checkpoint': plan['checkpoint']}}))
    result = subprocess.run([
        sys.executable, '-m', 'roboego_hand_vis.hand.run', '--config', str(config),
        '--input_dir', plan['input_dir'], '--output_dir', plan['output_dir']],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
    assert result.returncode == mint_exit, result.stderr
    output = Path(plan['output_dir'])
    report = json.loads((output / 'run_report.json').read_text())
    assert report['returncode'] == mint_exit
    assert (output / 'ego_called').exists() == (mint_exit == 0)
    if mint_exit == 0:
        assert (output / 'ego_called').read_text().strip() == str(ego)
