"""Shared implementation and dataset adapter boundaries."""
import importlib
from pathlib import Path
import pytest
from unittest.mock import patch

from roboego_hand_vis.lerobot_fast_foundation_depth.L1 import compute as l1
from roboego_hand_vis.lerobot_fast_foundation_depth.L3 import compute as l3
from roboego_hand_vis.lerobot_fast_foundation_depth.base import compute as shared


@pytest.mark.parametrize('level', ['L1', 'L3'])
def test_cli_iteration_override(level):
    cli = importlib.import_module(f'roboego_hand_vis.lerobot_fast_foundation_depth.{level}.run')
    with patch.object(cli, 'discover', return_value=[Path('session')]), \
         patch.object(cli, 'plan_session', return_value=[{'episode': 'episode'}]), \
         patch('roboego_hand_vis.lerobot_fast_foundation_depth.base.throughput.run_jobs', return_value=[]) as jobs:
        assert cli.main(['--source', 'session', '--valid-iters', '16']) == 0
        assert jobs.call_args.args[2]['valid_iters'] == 16
    with pytest.raises(SystemExit):
        cli.main(['--source', 'session', '--valid-iters', '0'])


def test_old_imports_resolve_to_shared_modules():
    prefix = 'roboego_hand_vis.lerobot_fast_foundation_depth'
    for name in ('engine', 'throughput', 'pipeline', 'depth_video',
                 'feature_storage', 'video_storage', 'reuse',
                 'model_compat', 'visualization'):
        assert importlib.import_module(f'{prefix}.L1.{name}') is importlib.import_module(f'{prefix}.base.{name}')


def test_adapters_forward_dataset_config_and_cameras():
    for adapter in (l1, l3):
        with patch.object(shared, 'run_session', return_value=[]) as run:
            assert adapter.run_session('session') == []
            run.assert_called_once_with('session', config=adapter.DEFAULT_CONFIG,
                                        camera_names=adapter.CAMERAS)
        assert adapter.DEFAULT_CONFIG.parent.name in ('L1', 'L3')


def test_adapters_preserve_explicit_config():
    for adapter in (l1, l3):
        with patch.object(shared, 'run_session') as run:
            adapter.run_session('session', config='custom.yaml', start=10)
            run.assert_called_once_with('session', config='custom.yaml', start=10,
                                        camera_names=adapter.CAMERAS)
