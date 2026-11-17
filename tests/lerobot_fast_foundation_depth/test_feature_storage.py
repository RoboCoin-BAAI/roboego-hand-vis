import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from roboego_hand_vis.lerobot_fast_foundation_depth.L1.feature_storage import publish_feature, depth_feature
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.reuse import existing_valid


FEATURE = 'observation.images.depth_headband_left_rectified'


def prepare(tmp_path, index, value=b'depth-video'):
    episode = tmp_path / 'data/chunk-000' / f'episode_{index:06d}'
    episode.mkdir(parents=True, exist_ok=True)
    timestamps = pa.table({'timestamp': [0., .033], 'episode_index': [index, index]})
    pq.write_table(timestamps, episode / 'observation.images.headband_left_rgb_rectified.parquet')
    work = tmp_path / f'work-{index}'
    work.mkdir()
    pq.write_table(timestamps, work / 'source_timestamps.parquet')
    (work / 'depth_hevc12.mkv').write_bytes(value)
    (work / 'lerobot_depth.json').write_text(json.dumps({
        'processed_frames': [0, 1], 'depth_storage': {'file': 'depth_hevc12.mkv', 'frames': 2, 'width': 10, 'height': 10}}))
    return episode, work


def test_single_episode_publication_and_overwrite(tmp_path):
    output = tmp_path / 'depth/chunk-000' / FEATURE
    ep, work = prepare(tmp_path, 0)
    publish_feature(work, output, ep, FEATURE)
    report = json.loads((output/'lerobot_depth.json').read_text())
    assert 'episodes' not in report
    assert report['depth_storage']['file'] == 'episode_000000.mkv'
    assert existing_valid({'episode': str(ep), 'output': str(output)}, verify_video=False)
    ep, work = prepare(tmp_path, 0, b'replacement')
    publish_feature(work, output, ep, FEATURE)
    assert (output/'episode_000000.mkv').read_bytes() == b'replacement'
    assert pq.read_table(output/(FEATURE+'.parquet')).num_rows == 2
    ep, work = prepare(tmp_path, 1)
    with pytest.raises(ValueError, match='another episode'):
        publish_feature(work, output, ep, FEATURE)
    assert (output/'episode_000000.mkv').read_bytes() == b'replacement'


def test_legacy_single_episode_manifest_is_flattened_on_publication(tmp_path):
    output = tmp_path / 'depth/chunk-000' / FEATURE
    ep, work = prepare(tmp_path, 0)
    publish_feature(work, output, ep, FEATURE)
    path = output/'lerobot_depth.json'
    report = json.loads(path.read_text())
    path.write_text(json.dumps({'schema_version': 2, 'depth_feature': FEATURE,
                                'episodes': {ep.name: report}}))
    assert existing_valid({'episode': str(ep), 'output': str(output)}, verify_video=False)
    ep, work = prepare(tmp_path, 0)
    publish_feature(work, output, ep, FEATURE)
    assert 'episodes' not in json.loads(path.read_text())


def test_l1_layout_unchanged_and_l3_feature_name():
    assert depth_feature({'left': 'head_left_rectified'}) is None
    assert depth_feature({'left': 'headband_left_rgb_rectified'}) == FEATURE


def test_invalid_episode_timestamps_do_not_replace_published_output(tmp_path):
    ep, work = prepare(tmp_path, 0)
    output = tmp_path / 'depth/chunk-000' / FEATURE
    publish_feature(work, output, ep, FEATURE)
    before = (output/'lerobot_depth.json').read_bytes()
    ep, work = prepare(tmp_path, 0, b'bad')
    pq.write_table(pa.table({'timestamp': [0.], 'episode_index': [99]}), work/'source_timestamps.parquet')
    with pytest.raises(ValueError, match='episode_index'):
        publish_feature(work, output, ep, FEATURE)
    assert (output/'lerobot_depth.json').read_bytes() == before
    assert (output/'episode_000000.mkv').read_bytes() == b'depth-video'
