import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'roboego_hand_vis/vio/deploy/vio-run.sh'


@pytest.mark.parametrize('exit_code', [0, 7])
def test_direct_runner_forwards_parameters_and_preserves_logs(tmp_path, exit_code):
    rootfs = tmp_path / 'rootfs'
    (rootfs / 'dev').mkdir(parents=True)
    for name in ('null', 'zero', 'random', 'urandom'):
        (rootfs / 'dev' / name).touch()
    session = tmp_path / 'session with spaces'
    session.mkdir()
    (session / 'input').write_text('original')
    output = tmp_path / 'output'
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    chroot = binaries / 'chroot'
    chroot.write_text('''#!/bin/bash
root=$1
printf '%s\\n' "$@" > "$root/arguments"
printf '{"native":true}\\n' > "$root/data/output/report.json"
echo 'native console details'
exit ''' + str(exit_code) + '\n')
    chroot.chmod(0o755)
    env = dict(os.environ, VIO_ROOTFS=str(rootfs), PATH=str(binaries)+':'+os.environ['PATH'])
    result = subprocess.run(['bash', str(SCRIPT), str(session), str(output), 'run',
                             '--input_mode', 'rectified', '--max_imu_boundary_gap_s', '0.3'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    args = (rootfs / 'arguments').read_text()
    assert 'runtime-output.py' not in args and '/usr/local/bin/vio-runtime' not in args
    assert 'roboego_hand_vis.vio.run' in args
    assert '--max_imu_boundary_gap_s\n0.3\n' in args
    assert '--output_dir\n/data/output\n' in args
    assert 'OPENVINS_TIME_OFFSET_MODE=estimate' in args
    assert (output / 'report.json').read_text() == '{"native":true}\n'
    assert 'native console details' in (output / 'console.log').read_text()
    if exit_code:
        assert (output / 'error.log').read_text() == (output / 'console.log').read_text()
    assert (session / 'input').read_text() == 'original'
    assert not (rootfs / str(session).lstrip('/')).exists()
    # Never overwrite a previous output on retry.
    again = subprocess.run(['bash', str(SCRIPT), str(session), str(output), 'run'],
                           env=env, capture_output=True, text=True)
    assert again.returncode != 0
    assert (output / 'report.json').read_text() == '{"native":true}\n'


@pytest.mark.parametrize('collision', ['input', 'output', 'symlink'])
def test_refuses_existing_rootfs_data(tmp_path, collision):
    rootfs = tmp_path / 'rootfs'
    rootfs.mkdir()
    session = tmp_path / 'session'
    session.mkdir()
    staged = rootfs / str(session).lstrip('/')
    if collision == 'input':
        staged.mkdir(parents=True)
        protected = staged / 'keep'
    elif collision == 'output':
        (rootfs / 'data/output').mkdir(parents=True)
        protected = rootfs / 'data/output/keep'
    else:
        outside = tmp_path / 'outside'
        outside.mkdir()
        (rootfs / 'data').symlink_to(outside, target_is_directory=True)
        protected = outside / 'keep'
    protected.write_text('must survive')
    result = subprocess.run(['bash', str(SCRIPT), str(session), str(tmp_path / 'out'), 'run'],
                            env=dict(os.environ, VIO_ROOTFS=str(rootfs)),
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert protected.read_text() == 'must survive'
    assert 'already exists' in result.stderr or 'not empty' in result.stderr or 'symlink' in result.stderr


def test_rechecks_output_after_waiting_for_lock(tmp_path):
    rootfs = tmp_path / 'rootfs'
    rootfs.mkdir()
    session = tmp_path / 'session'
    session.mkdir()
    output = tmp_path / 'out'
    output.mkdir()
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    # Model a preceding lock holder publishing its result before lock acquisition returns.
    flock = binaries / 'flock'
    flock.write_text('#!/bin/bash\nprintf previous > "$TEST_OUTPUT/previous"\n')
    flock.chmod(0o755)
    result = subprocess.run(['bash', str(SCRIPT), str(session), str(output), 'run'],
                            env=dict(os.environ, VIO_ROOTFS=str(rootfs), TEST_OUTPUT=str(output),
                                     PATH=str(binaries)+':'+os.environ['PATH']),
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Output directory must be empty' in result.stderr
    assert (output / 'previous').read_text() == 'previous'
    assert not (rootfs / str(session).lstrip('/')).exists()
