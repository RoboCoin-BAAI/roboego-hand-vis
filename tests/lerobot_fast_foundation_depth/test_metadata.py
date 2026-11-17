import json
from unittest.mock import patch

import pytest

from roboego_hand_vis.lerobot_fast_foundation_depth.base.metadata import video_feature_definition
from roboego_hand_vis.lerobot_fast_foundation_depth.base.video_storage import publish_video


def test_publication_registers_depth_and_preserves_session_info(tmp_path):
    session = tmp_path
    meta = session / 'meta/info.json'
    meta.parent.mkdir()
    original = {'features': {'rgb': {'dtype': 'video'}}, 'total_videos': 1,
                'total_frames': 123, 'robot_type': 'L3'}
    meta.write_text(json.dumps(original))
    rgb = session / 'videos/chunk-000/rgb/episode_000000.mp4'
    rgb.parent.mkdir(parents=True)
    rgb.write_bytes(b'rgb')
    camera = 'observation.images.headband_left_rgb_rectified'
    feature = 'observation.images.depth_headband_left_rectified'
    storage = dict(width=64, height=48, fps=30, codec='hevc',
                   pixel_format='gray12le', container='mp4', unit='millimeter',
                   invalid_value=0, max_depth_m=4., lossless=False,
                   requires_valid_mask=True)

    def publish(index):
        episode = session / 'data/chunk-000' / f'episode_{index:06d}'
        episode.mkdir(parents=True, exist_ok=True)
        work = session / 'work'
        work.mkdir()
        for name in ('depth_hevc12.mp4', 'valid_mask_ffv1.mkv', 'source_timestamps.parquet'):
            (work / name).write_bytes(b'asset')
        (work / 'lerobot_depth.json').write_text(json.dumps({'depth_storage': storage}))
        output = session / 'annotation/chunk-000' / feature / episode.name
        publish_video(work, output, session, episode, camera)

    publish(0)
    first = json.loads(meta.read_text())
    assert first['features'][feature] == video_feature_definition(storage)
    assert first['features']['rgb'] == original['features']['rgb']
    assert first['total_frames'] == 123
    assert first['robot_type'] == 'L3'
    assert first['total_videos'] == 2
    assert not list(session.rglob('*.lock'))
    publish(0)
    assert json.loads(meta.read_text())['total_videos'] == 2
    publish(1)
    assert json.loads(meta.read_text())['total_videos'] == 3

    previous = meta.read_bytes()
    with patch('roboego_hand_vis.lerobot_stereo_rectification.L1.compute.publish_files',
               side_effect=OSError('publication failed')):
        with pytest.raises(OSError, match='publication failed'):
            publish(2)
    assert meta.read_bytes() == previous
    assert not (session / 'videos/chunk-000' / feature / 'episode_000002.mp4').exists()
