import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from roboego_hand_vis.lerobot_fast_foundation_depth.L1.depth_video import HEVC12Writer, read_episode_depth
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.video_storage import publish_video
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.reuse import existing_valid


@pytest.mark.parametrize('camera', ['head_left_rectified', 'headband_left_rgb_rectified'])
def test_compressed_mp4_mask_publication_and_reuse(tmp_path, camera):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    episode = tmp_path / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    feature = 'observation.images.' + camera
    times = pa.table({'timestamp': [0., .033]})
    pq.write_table(times, episode / (feature + '.parquet'))
    work = tmp_path / 'work'
    work.mkdir()
    pq.write_table(times, work / 'source_timestamps.parquet')
    depth = np.linspace(.001, 4., 4096, dtype=np.float32).reshape(64, 64)
    depth[::2, ::3] = np.nan
    writer = HEVC12Writer(work / 'depth_hevc12.mp4', size=(64, 64), fps=30, ffmpeg=ffmpeg)
    try:
        writer.write(depth)
        writer.write(depth[::-1].copy())
    finally:
        writer.close()
    report = {'processed_frames': [0, 1], 'depth_storage': {
        'file': 'depth_hevc12.mp4', 'mask_file': 'valid_mask_ffv1.mkv',
        'requires_valid_mask': True, 'lossless': False, 'frames': 2, 'width': 64, 'height': 64}}
    (work / 'lerobot_depth.json').write_text(json.dumps(report))
    output = tmp_path / 'annotation/chunk-000' / feature.replace('observation.images.', 'observation.images.depth_').replace('_rgb_rectified', '_rectified') / 'episode_000000'
    publish_video(work, output, tmp_path, episode, feature)
    decoded = list(read_episode_depth(output, ffmpeg=ffmpeg))
    assert len(decoded) == 2
    for original, restored in zip((depth, depth[::-1]), decoded):
        np.testing.assert_array_equal(np.isnan(original), np.isnan(restored))
    plan = {'episode': str(episode), 'output': str(output)}
    assert existing_valid(plan, verify_video=False)
    storage = json.loads((output / 'lerobot_depth.json').read_text())['depth_storage']
    metadata = json.loads((output / 'lerobot_depth.json').read_text())
    assert metadata['timestamp_file'].startswith('data/chunk-000/episode_000000/')
    assert pq.read_table(tmp_path / metadata['timestamp_file']).equals(times)
    assert not (tmp_path / 'depth').exists()
    assert 'depth' not in Path(storage['mask_file']).parts
    assert storage['file'].startswith('videos/')
    assert storage['mask_file'].startswith('annotation/')
    (tmp_path / storage['mask_file']).unlink()
    assert not existing_valid(plan, verify_video=False)
    with pytest.raises(FileNotFoundError):
        list(read_episode_depth(output, ffmpeg=ffmpeg))
