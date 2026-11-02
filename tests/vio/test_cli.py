from __future__ import annotations

from roboego_hand_vis.vio import cli


def test_cli_overrides_imu_boundary_gap() -> None:
    args = cli.parse_args(
        [
            "--session",
            "/tmp/session",
            "--max_imu_boundary_gap_s",
            "0.3",
        ]
    )

    config = cli.config_from_yaml_and_args(
        {
            "prepare": {"max_imu_boundary_gap_s": 0.1},
            "io": {"output_dir": "{session}/vio"},
            "openvins": {
                "root": "/tmp/open_vins",
                "binary": "{openvins_root}/ov_msckf/build/roboego_openvins_offline",
            },
            "runtime": {"system_python": "/usr/bin/python3"},
        },
        args,
    )

    assert config.max_imu_boundary_gap_s == 0.3


def test_native_command_exports_public_delivery_after_offline_runner():
    from pathlib import Path
    from roboego_hand_vis.vio.runner import build_openvins_command
    args = cli.parse_args(['--session', '/tmp/session'])
    config = cli.config_from_yaml_and_args({}, args)
    command = build_openvins_command(config, stage_dir=Path('/tmp/stage'))
    import shlex
    lines = [shlex.split(line) for line in command.splitlines()]
    prepare = next(parts for parts in lines if any(p.endswith('prepare_entry.py') for p in parts))
    export = next(parts for parts in lines if any(p.endswith('export_entry.py') for p in parts))
    work = str(config.output_dir / '.work' / 'run')
    assert prepare[prepare.index('--output') + 1] == work
    assert export[export.index('--work') + 1] == work
    assert export[export.index('--output') + 1] == str(config.output_dir)
    assert command.index('offline_entry.py') < command.index('export_entry.py')


def test_main_logs_effective_boundary_override(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from roboego_hand_vis.vio import cli
    config = tmp_path / 'config.yaml'
    config.write_text('prepare:\n  max_imu_boundary_gap_s: 0.1\n')
    monkeypatch.setattr(sys, 'argv', ['vio', '--session', str(tmp_path),
                                     '--config', str(config),
                                     '--max_imu_boundary_gap_s', '0.3'])
    received = []
    monkeypatch.setattr(cli, 'run_openvins', received.append)
    cli.main()
    line = capsys.readouterr().out.strip()
    assert line.startswith('VIO effective config: ')
    settings = json.loads(line.removeprefix('VIO effective config: '))
    assert settings['max_imu_boundary_gap_s'] == 0.3
    assert settings['session_dir'] == str(tmp_path)
    assert received[0].max_imu_boundary_gap_s == 0.3


def test_estimator_cli_overrides_yaml_and_reaches_commands(tmp_path):
    from pathlib import Path
    from roboego_hand_vis.vio.runner import build_openvins_command
    args = cli.parse_args(['--session', str(tmp_path), '--zupt',
                           '--time-offset-mode', 'config', '--timing-bootstrap', 'off',
                           '--init-window-time', '3', '--no-init-dyn-use'])
    config = cli.config_from_yaml_and_args({'run': {'zupt': False, 'init_window_time': 2}}, args)
    assert config.zupt and config.init_window_time == 3 and not config.init_dyn_use
    command = build_openvins_command(config, stage_dir=Path('/stage'))
    assert '--zupt --no-init-dyn-use' in command
    assert '--init-window-time 3.0' in command
    assert '--time-offset-mode config --timing-bootstrap off' in command


def test_offline_time_options_reach_launcher_environment(tmp_path, monkeypatch):
    import json
    import sys
    from roboego_hand_vis.vio import offline_entry
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(output_dir=str(tmp_path), estimator_config_path='config.yaml',
        videos={'left': {'path': 'left.mkv'}, 'right': {'path': 'right.mkv'}},
        stereo_pairs_path='pairs.csv', imu_clean_path='imu.csv')))
    monkeypatch.setattr(sys, 'argv', ['offline', '--binary', 'launcher', '--manifest', str(manifest),
                                     '--time-offset-mode', 'config', '--timing-bootstrap', 'off'])
    received = []
    monkeypatch.setattr(offline_entry.subprocess, 'run', lambda *a, **kw: received.append(kw))
    offline_entry.main()
    assert received[0]['env']['OPENVINS_TIME_OFFSET_MODE'] == 'config'
    assert received[0]['env']['OPENVINS_TIMING_BOOTSTRAP'] == 'off'
