import json

from roboego_hand_vis.annotation_statistics.assign_common_tasks import assign_document
from roboego_hand_vis.annotation_statistics.clean_actions import clean_document
from roboego_hand_vis.annotation_statistics.readme_sync import prepare_readme, write_readme


def test_existing_readme_actions_cleaned_without_touching_other_metadata(tmp_path):
    annotation = tmp_path/'annotation/episode_000000_structured.json'
    annotation.parent.mkdir()
    (tmp_path/'meta').mkdir()
    readme = tmp_path/'meta/readme.json'
    readme.write_text(json.dumps({'actions': ['PUT,', 'continue', 'place'],
                                 'task_name': 'original', 'frame_count': 42}))
    rules = {'delete': {'continue': ''}, 'replace': {'put': 'place'},
             'normalize': {'lowercase': True, 'trim_whitespace_and_edge_punctuation': True}}
    write_readme(prepare_readme(annotation, clean_document, rules, 'actions'))
    assert json.loads(readme.read_text()) == {'actions': ['place'], 'task_name': 'original', 'frame_count': 42}
    assert prepare_readme(annotation, clean_document, rules, 'actions') is None


def test_missing_readme_not_created(tmp_path):
    assert prepare_readme(tmp_path/'annotation/episode_000000_structured.json', clean_document,
                          {'delete': {}}, 'actions') is None


def test_common_task_is_additive_and_unmatched_is_null():
    source = {'task_name': 'Fold blue towels', 'subtask': [{'action': ['fold']}]}
    result = assign_document(source, {'Fold blue towels': 'organize household linen'})
    assert result == {**source, 'common_task': 'organize household linen'}
    assert 'common_task' not in source
    assert assign_document(source, {})['common_task'] is None


def test_readme_objects_and_scene_keep_attributes_and_metadata(tmp_path):
    from roboego_hand_vis.annotation_statistics.clean_objects import clean_document as clean_objects
    from roboego_hand_vis.annotation_statistics.clean_scenes import clean_document as clean_scenes
    from pathlib import Path
    rules_dir = Path(__file__).parents[1]/'roboego_hand_vis/annotation_statistics'
    annotation = tmp_path/'annotation/episode_000000_structured.json'
    (tmp_path/'meta').mkdir()
    readme = tmp_path/'meta/readme.json'
    original = {'objects': [{'name': 'Blue cups', 'attributes': {'color': 'blue'}}],
                'scene': {'type': 'indoor', 'category': 'commercial_service', 'name': 'office_area'},
                'common_task': 'organize container', 'frame_count': 42}
    readme.write_text(json.dumps(original))
    for cleaner, rulefile, field in [(clean_objects, 'object_rules.json', 'objects'),
                                     (clean_scenes, 'scene_rules.json', 'scene')]:
        rules = json.loads((rules_dir/rulefile).read_text(encoding='utf-8'))
        write_readme(prepare_readme(annotation, cleaner, rules, field))
    result = json.loads(readme.read_text())
    assert result['objects'] == [{'name': 'cup', 'attributes': {'color': 'blue'}}]
    assert result['scene'] == {'type': 'indoor', 'category': 'professional_work', 'name': 'office'}
    assert result['common_task'] == original['common_task']
    assert result['frame_count'] == 42
