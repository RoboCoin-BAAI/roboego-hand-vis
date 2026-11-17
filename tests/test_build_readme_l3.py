import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from roboego_hand_vis.build_json.L3 import build_readme as l3
from roboego_hand_vis.build_json.base import build_readme as base


def session_fixture(root):
    (root / 'meta').mkdir()
    (root / 'meta/info.json').write_text(json.dumps(dict(
        features={'observation.hand_skeleton': dict(shape=[50, 6], fps=30)},
        fps=30, total_frames=999)), encoding='utf-8')
    data = root / 'data/chunk-000/episode_000000'
    data.mkdir(parents=True)
    pq.write_table(pa.table({'timestamp': [0., 1 / 30]}), data / 'master_index.parquet')
    pq.write_table(pa.table({'value': [1., 2.]}), data / 'observation.hand_skeleton.parquet')
    (root / 'annotation').mkdir()
    (root / 'annotation/episode_000000_structured.json').write_text(json.dumps(dict(
        task_name='Cut hair', common_task='Groom hair', task_description='Trim hair',
        scene_classification=dict(category='commercial_service', area='hair_salon'),
        objects=[], subtask=[dict(action=['untrusted_fallback'])], actions=dict(vocabulary=[]))),
        encoding='utf-8')
    return root


@pytest.mark.parametrize('version', [1, 2])
def test_l3_depth_and_shared_fields(tmp_path, version):
    session = session_fixture(tmp_path)
    folder = session / 'depth/chunk-000/observation.images.depth_headband_left_rectified'
    folder.mkdir(parents=True)
    (folder / 'episode_000000.mkv').write_bytes(b'depth')
    entry = dict(depth_storage=dict(file='episode_000000.mkv', height=12, width=16,
        frames=2, fps=30, unit='m', invalid_value=0, pixel_format='gray12le'))
    missing = dict(depth_storage={**entry['depth_storage'], 'file': 'missing.mkv'})
    report = dict(schema_version=2, episodes={'0': entry, '1': missing}) if version == 2 else entry
    (folder / 'lerobot_depth.json').write_text(json.dumps(report), encoding='utf-8')
    result = l3.build(session)
    assert result['scene'] == dict(type='indoor', category='commercial_service', name='hair_salon')
    assert result['common_task'] == 'groom hair'
    assert result['actions'] == []
    assert result['frame_count'] == 2
    assert result['duration_s'] == 2 / 30
    assert result['hand_capture_type'] == 'glove'
    assert result['modalities']['glove']['streams'][0]['sample_count'] == 2
    depth = result['modalities']['depth']['streams']
    assert len(depth) == 1
    assert depth[0]['format'] == 'hevc12'
    assert depth[0]['shape'] == [12, 16, 1]
    assert depth[0]['sample_count'] == 2
    assert base.build(session)['modalities']['depth']['streams'] == depth
    (session / 'meta/readme.json').write_text(json.dumps(result), encoding='utf-8')
    assert l3.build(session) == result


def test_l3_does_not_fill_semantics_from_collection(tmp_path):
    session = session_fixture(tmp_path)
    (session / 'annotation/episode_000000_structured.json').unlink()
    (session / 'meta/collect_info.jsonl').write_text(json.dumps(dict(scene='warehouse')))
    result = l3.build(session, dict(task_name='Invented task', scene=dict(name='warehouse')))
    assert result['task_name'] is None
    assert result['common_task'] is None
    assert result['scene'] == dict(type='unknown', category=None, name=None)
    assert 'depth' not in result['modalities']
