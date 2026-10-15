"""Adapter for vio-run.sh SESSION OUTPUT run [VIO CLI options]."""

from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from roboego_hand_vis.common.processes import run_logged_command


def validate_runner(path):
    script = Path(path).expanduser().resolve()
    if not script.is_file() or not os.access(script, os.R_OK):
        raise ValueError(f'Chroot runner is missing or unreadable: {script}')
    if shutil.which('flock') is None:
        raise ValueError('Chroot runner requires flock on the host')
    return script


def run_chroot_session(config, log, runner, *, check_outputs):
    script = validate_runner(runner)
    temporary = Path(tempfile.mkdtemp(prefix='.vio-chroot-', dir=config.session_dir))
    # Keep failed output for diagnosis. Never allow old success to mask a failed retry.
    command = ['flock', '--exclusive', str(script) + '.database.lock',
               'bash', str(script), str(config.session_dir), str(temporary),
               'run', '--input_mode', config.input_mode,
               '--start_s', str(config.start_s),
               '--stereo_tolerance_s', str(config.stereo_tolerance_s),
               '--max_imu_gap_s', str(config.max_imu_gap_s),
               '--max_imu_boundary_gap_s', str(config.max_imu_boundary_gap_s)]
    if config.duration_s is not None:
        command += ['--duration_s', str(config.duration_s)]
    if not config.downsample_cameras:
        command.append('--no-downsample_cameras')
    try:
        run_logged_command(command, log, cwd=Path(__file__).resolve().parents[2])
        if any(path.is_symlink() for path in temporary.rglob('*')):
            raise ValueError('Chroot output must not contain symbolic links')
        staged_config = replace(config, output_dir=temporary)
        fresh = {temporary / name: None for name in
                 ('report.json', 'trajectory.npz')}
        check_outputs(staged_config, fresh)
        backup = None
        if config.output_dir.exists():
            backup = config.session_dir / ('.vio-previous-' + uuid.uuid4().hex)
            config.output_dir.rename(backup)
        try:
            temporary.rename(config.output_dir)
        except BaseException:
            if backup is not None:
                backup.rename(config.output_dir)
            raise
    except Exception as exc:
        raise RuntimeError(f'{exc}; chroot output retained at {temporary}') from exc
