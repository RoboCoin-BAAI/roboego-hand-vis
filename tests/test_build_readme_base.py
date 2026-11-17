import json
import pytest
from roboego_hand_vis.build_json.base import build_readme as base
from roboego_hand_vis.build_json.guanglun import build_readme as old


def test_legacy_entry_delegates():
    assert old.build is base.build
    assert old.main is base.main


def test_generic_session_without_guanglun_metadata(tmp_path):
    (tmp_path/'meta').mkdir()
    (tmp_path/'meta/info.json').write_text(json.dumps(dict(features={},fps=30,total_frames=60)))
    (tmp_path/'annotation').mkdir()
    (tmp_path/'annotation/episode_000000_structured.json').write_text(json.dumps(dict(
        task_name='Cut hair',common_task='groom hair',task_description='Trim hair',scene_classification=dict(
            category='commercial_service',area='hair_salon'),objects=[],subtask=[],actions=dict(vocabulary=['trim']))))
    result=base.build(tmp_path)
    assert result['scene']==dict(type='indoor',category='commercial_service',name='hair_salon')
    assert result['frame_count']==60
    assert result['duration_s']==2
    assert result['actions']==['trim']
    assert result['common_task']=='groom hair'
    assert base.main(['--root',str(tmp_path)])==0
    first=json.loads((tmp_path/'meta/readme.json').read_text())
    assert base.main(['--root',str(tmp_path)])==0
    assert json.loads((tmp_path/'meta/readme.json').read_text())==first
    assert not list((tmp_path/'meta').glob('.readme.json.*'))


@pytest.mark.parametrize('actions', [{}, {'actions': {'vocabulary': []}}])
def test_empty_vocabulary_does_not_fall_back_to_subtask(tmp_path, actions):
    (tmp_path/'meta').mkdir()
    (tmp_path/'meta/info.json').write_text(json.dumps(dict(features={}, fps=30, total_frames=60)))
    (tmp_path/'annotation').mkdir()
    annotation = dict(objects=[], subtask=[{'action': ['continue'], 'labels': ['use']}], **actions)
    (tmp_path/'annotation/episode_000000_structured.json').write_text(json.dumps(annotation))
    assert base.build(tmp_path)['actions'] == []
