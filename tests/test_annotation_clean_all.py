import json
from pathlib import Path
from unittest.mock import patch

from roboego_hand_vis.annotation_statistics import clean_all as module


def test_single_discovery_and_one_read_write_per_file(tmp_path):
    session = tmp_path/'dataset/batch_0000001/session'
    ann = session/'annotation/episode_000000_structured.json'
    readme = session/'meta/readme.json'
    ann.parent.mkdir(parents=True)
    readme.parent.mkdir()
    source = {'task_name': 'Organize files', 'subtask': [{'action': ['continue']}],
              'actions': {'vocabulary': ['PUT,', 'continue']},
              'objects': [{'name': 'Blue cups'}],
              'scene_classification': {'category': 'commercial_service', 'area': 'office_area'}}
    ann.write_text(json.dumps(source))
    readme.write_text(json.dumps({'actions': ['PUT,', 'continue'], 'objects': [{'name': 'Blue cups'}],
                                 'scene': {'category': 'commercial_service', 'name': 'office_area'},
                                 'task_name': 'organize_files', 'frame_count': 123}))
    rules_dir = Path(module.__file__).parent
    rules = {key: module.load_object(rules_dir/name) for key, name in [
        ('actions', 'nonverb_rules.json'), ('objects', 'object_rules.json'), ('scene', 'scene_rules.json')]}
    with patch.object(module, 'find_annotations', wraps=module.find_annotations) as discover, \
         patch.object(module, 'load_object', wraps=module.load_object) as read, \
         patch.object(module, 'write_object', wraps=module.write_object) as write, \
         patch.object(Path, 'rglob', side_effect=AssertionError('Recursive scan forbidden')):
        stats = module.clean_all(tmp_path/'dataset', {'Organize files': 'organize document'}, rules)
        assert discover.call_count == 1
        assert read.call_count == 2
        assert write.call_count == 2
    assert stats['failed_sessions'] == 0
    cleaned = json.loads(ann.read_text())
    result = json.loads(readme.read_text())
    assert cleaned['actions']['vocabulary'] == result['actions'] == ['place']
    assert cleaned['objects'] == result['objects'] == [{'name': 'cup'}]
    assert cleaned['subtask'] == source['subtask']
    assert cleaned['task_name'] == source['task_name']
    assert result['frame_count'] == 123
    assert result['scene']['category'] == 'professional_work'
    assert result['common_task'] == cleaned['common_task'] == 'organize document'
    again = module.clean_all(tmp_path/'dataset', {'Organize files': 'organize document'}, rules)
    assert again['annotations_modified'] == again['readmes_modified'] == 0
