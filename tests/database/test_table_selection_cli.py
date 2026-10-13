import os
import subprocess
import sys

import pytest

STAGES = ['raw_headband_integrity', 'video_simple_quality', 'detect_hand',
          'stereo_rectification', 'qwen_annotation', 'post_annotation_actions',
          'stereo_depth_gate', 'fast_foundation_depth', 'hand', 'archive_depth_cache',
          'vio', 'video_privacy_blur', 'download_data.l1']


@pytest.mark.parametrize('stage', STAGES)
def test_database_entry_exposes_table_selection_without_connection(stage):
    module = 'roboego_hand_vis.' + stage
    if stage != 'download_data.l1':
        module += '.database_run'
    env = dict(os.environ, DATABASE_URL='invalid-url')
    result = subprocess.run([sys.executable, '-m', module, '--help'],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert '--table {l1data,l2data,l3data}' in result.stdout
    if stage != 'download_data.l1':
        assert '--local-root' in result.stdout
