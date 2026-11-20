import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / 'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('task_name_cleaner', DIRECTORY / 'regenerate_task_names_qwen.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / 'dataset'
    path = root / 'batch_0000001/sample/annotation/episode_000000_structured.json'
    path.parent.mkdir(parents=True)
    raw = (b'\xef\xbb\xbf{\r\n"task_name":"trim hair", "objects":[{"name":"hair","attributes":{"color":"black"}}],'
           b'"actions":{"vocabulary":["trim","place"]},"task_description":"Give a customer a haircut",'
           b'"original_description":"Cut hair", "subtask":[{"action":"trim","start":1.00}],'
           b'"scene_classification":{"category":"other","area":"other"},"keep":1.00\r\n}')
    path.write_bytes(raw)
    source = m.evidence(json.loads(raw.decode('utf-8-sig')))
    record = dict(path=path.relative_to(root).as_posix(), status='done', source=source,
                  before='trim hair', after='cut hair', decision=dict(verb='cut', object_name='hair', object_indices=[0], reason='Haircut'))
    return root.resolve(), path, raw, record


def test_only_subtask_verbs_and_existing_main_objects(fixture):
    source = fixture[3]['source']
    assert m.validate(dict(verb='cut', object_name='hair', object_indices=[0], reason='Haircut'), source, {'cut'}) == 'cut hair'
    for result in (dict(verb='pick', object_name='hair', object_indices=[0], reason='Atomic'),
                   dict(verb='cut', object_name='hair', object_indices=[True], reason='Boolean index'),
                   dict(verb='cut', object_name='banana', object_indices=[1], reason='New object')):
        with pytest.raises(ValueError):
            m.validate(result, source, {'cut'})
    assert m.validate(dict(verb=None, object_name=None, object_indices=[], reason='Unclear'), source, {'cut'}) is None


def test_apply_changes_only_task_name_and_keeps_latest_scene(fixture, tmp_path):
    root, path, raw, record = fixture
    latest = raw.replace(b'"category":"other","area":"other"', b'"category":"household","area":"bedroom"')
    path.write_bytes(latest)
    result = m.apply_one(root, record, tmp_path / 'report', {'cut'})
    assert result['changed']
    assert path.read_bytes() == latest.replace(b'"task_name":"trim hair"', b'"task_name":"cut hair"')
    assert (tmp_path / 'report/original_annotations' / record['path']).read_bytes() == latest
    assert not m.apply_one(root, record, tmp_path / 'report', {'cut'})['changed']


@pytest.mark.parametrize('old,new', [(b'"name":"hair"', b'"name":"customer hair"'),
                                      (b'["trim","place"]', b'["cut"]'),
                                      (b'Cut hair', b'Wash hair')])
def test_apply_refuses_changed_labels_or_evidence(fixture, tmp_path, old, new):
    root, path, raw, record = fixture
    path.write_bytes(raw.replace(old, new))
    with pytest.raises(ValueError, match='evidence changed'):
        m.apply_one(root, record, tmp_path / 'report', {'cut'})


def test_paid_response_is_reused_and_prepare_never_writes(fixture, tmp_path, monkeypatch):
    root, path, raw, record = fixture
    args = SimpleNamespace(root=root, state_dir=tmp_path / 'state', model='qwen3.7-plus',
            max_tokens=1500, retry_failed=False, dry_run=False,
            overwrite=SimpleNamespace(job_lock=lambda p: nullcontext()))
    calls = []
    def fake(*a):
        calls.append(1)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(record['decision'])}}]}
    monkeypatch.setattr(m, 'request', fake)
    assert m.prepare_one(path, args, None, {'cut'})['after'] == 'cut hair'
    assert m.prepare_one(path, args, None, {'cut'})['after'] == 'cut hair'
    assert calls == [1]
    assert path.read_bytes() == raw


def test_dry_run_creates_preview_without_requesting_model(fixture, tmp_path, monkeypatch):
    root, path, raw, record = fixture
    def fail(*a):
        raise AssertionError('No API calls in preview')
    monkeypatch.setattr(m, 'request', fail)
    monkeypatch.setattr(m, 'review_modules', lambda p: (SimpleNamespace(settings=fail), SimpleNamespace(job_lock=lambda p: nullcontext())))
    output = tmp_path / 'names.json'
    assert m.main(['prepare', '--root', str(root), '--start', '1', '--end', '1',
                   '--output', str(output), '--argus-root', str(tmp_path), '--dry-run']) == 0
    assert json.loads(output.read_text())['status'] == 'preview'
    assert path.read_bytes() == raw


def test_modified_object_name_is_not_allowed_in_task_name(fixture):
    source = fixture[3]['source']
    source['object_names'] = ['small red cup']
    with pytest.raises(ValueError, match='without modifiers'):
        m.validate(dict(verb='wash', object_name='small red cup', object_indices=[0], reason='Wash cup'), source, {'wash'})


def test_whole_haircut_and_minor_finishing_have_distinct_names(fixture):
    source = fixture[3]['source']
    choice = dict(verb='trim', object_name='hair', object_indices=[0], reason='Haircut')
    assert m.validate(choice, source, {'cut', 'trim'}) == 'cut hair'
    source['task_description'] = 'Trim split ends only'
    assert m.validate(choice, source, {'cut', 'trim'}) == 'trim hair'


def test_umbrella_noun_supported_by_multiple_objects():
    source = dict(object_names=['hoodie', 'pants', 'shirt'], task_description='Organize wardrobe')
    choice = dict(verb='organize', object_name='clothing', object_indices=[0, 1, 2], reason='Multiple garments')
    assert m.validate(choice, source, {'organize'}) == 'organize clothing'


def test_unsupported_perming_goal_is_not_replaced_by_wrapping():
    source = dict(object_names=['hair'], task_description='Perm the hair on a mannequin')
    choice = dict(verb='wrap', object_name='hair', object_indices=[0], reason='Wrapping strands')
    assert m.validate(choice, source, {'wrap'}) is None


def test_identification_labels_and_decorative_stickers_remain_distinct():
    choice = dict(verb='stick', object_name='sticker', object_indices=[0], reason='Apply sticker')
    source = dict(object_names=['sticker'], task_description='Apply stickers to posts for marking')
    assert m.validate(choice, source, {'stick'}) == 'stick label'
    source['task_description'] = 'Apply decorative stickers to a notebook'
    assert m.validate(choice, source, {'stick'}) == 'stick sticker'


def test_redundant_object_in_verb_field_is_a_format_only_correction(fixture):
    source = fixture[3]['source']
    choice = dict(verb='cut hair', object_name='hair', object_indices=[0], reason='Haircut')
    assert m.validate(choice, source, {'cut'}) == 'cut hair'


def test_missing_task_name_can_be_added_without_rewriting_other_values(fixture, tmp_path):
    root, path, raw, record = fixture
    raw = raw.replace(b'"task_name":"trim hair", ', b'')
    path.write_bytes(raw)
    record.update(before=None, before_present=False)
    result = m.apply_one(root, record, tmp_path/'report', {'cut'})
    assert result['changed']
    after = path.read_bytes()
    assert after.startswith(b'\xef\xbb\xbf')
    assert b'"keep":1.00' in after
    expected = json.loads(raw.decode('utf-8-sig'))
    expected['task_name'] = 'cut hair'
    assert json.loads(after.decode('utf-8-sig')) == expected


def test_invalid_semantic_candidate_goes_to_review_and_reuses_cache(fixture, tmp_path, monkeypatch):
    root, path, raw, record = fixture
    args = SimpleNamespace(root=root, state_dir=tmp_path/'state', model='qwen3.7-plus',
            max_tokens=1500, retry_failed=False, dry_run=False,
            overwrite=SimpleNamespace(job_lock=lambda p: nullcontext()))
    calls = []
    def fake(*a):
        calls.append(1)
        result = dict(verb='roll', object_name='hair', object_indices=[0], reason='Unsupported verb')
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(result)}}]}
    monkeypatch.setattr(m,'request',fake)
    for _ in range(2):
        result = m.prepare_one(path,args,None,{'cut'})
        assert result['status']=='review' and result['after'] is None
        assert result['review_reason']
    assert calls==[1]
    assert path.read_bytes()==raw


def test_explicit_task_target_does_not_have_to_be_listed_as_an_object():
    source=dict(object_names=['cloth'],task_description='Wipe the table',subtask=[])
    choice=dict(verb='wipe',object_name='table',object_indices=[],reason='Explicit table target')
    assert m.validate(choice,source,{'wipe'})=='wipe table'


def test_duplicate_json_answers_require_same_verb_and_noun():
    result=dict(verb='cut',object_name='hair',object_indices=[0],reason='Haircut')
    payload={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(result)+json.dumps(dict(result,reason='Same purpose'))}}]}
    assert m.parse(payload)['verb']=='cut'
    payload['choices'][0]['message']['content']+=json.dumps(dict(result,verb='wash'))
    with pytest.raises(ValueError,match='Conflicting'):
        m.parse(payload)
