import json
from pathlib import Path
import sys

import pytest

BASE=Path(__file__).resolve().parents[2]/'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0,str(BASE))
import review_main_hand_tasks as m
RULES=json.loads((BASE/'task_name_normalization_rules.json').read_text())


def response(item):
    return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'items':[item]})}}]}


def test_one_hand_operation_and_supported_object():
    rows=[dict(old_task_name='Wipe boxes and put away tools',counts={'jianzhi':2},
               examples=[{'task_description':'Wipe the boxes.','objects':['box','cloth']}])]
    decision=dict(id=0,status='candidate',verb='wipe',object_name='box',source_noun='boxes',
                  evidence='Wipe the boxes.',reason='Main manipulation')
    assert m.validate(response(decision),rows,{'wipe','clean'},RULES)[0]['proposed']=='clean box'
    for edit in [dict(object_name='bathroom',source_noun='bathroom'),dict(object_name='box and shelf'),
                 dict(evidence='Invented text'),dict(verb='invented')]:
        assert m.validate(response(dict(decision,**edit)),rows,{'wipe','clean'},RULES)[0]['status']=='review'


def test_collects_all_distinct_contexts_not_only_first():
    records=[dict(dataset='jianzhi',task_name='organize box',objects=['box'],task_description='First context'),
             dict(dataset='guanglun',task_name='organize box',objects=['box'],task_description='Conflicting context')]
    result=m.make_inputs({'records':records})
    assert len(result)==1 and len(result[0]['examples'])==2
    assert result[0]['counts']=={'jianzhi':1,'guanglun':1}


def test_multitoken_object_is_one_concept_and_nouns_are_not_invented():
    assert m.canonical_noun('hair dryers',RULES)=='hair dryer'
    assert m.canonical_noun('glasses',RULES)=='glasses'


def test_one_unsupported_item_does_not_reject_its_packet():
    rows=[dict(old_task_name='wipe box',counts={'jianzhi':1},examples=[]),
          dict(old_task_name='wipe cup',counts={'guanglun':1},examples=[])]
    items=[dict(id=0,status='candidate',verb='wipe',object_name='box',source_noun='box',evidence='wipe box',reason='Supported'),
           dict(id=1,status='candidate',verb='wipe',object_name='bathroom',source_noun='cup',evidence='wipe cup',reason='Unsupported')]
    payload={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'items':items})}}]}
    results=m.validate(payload,rows,{'wipe','clean'},RULES)
    assert results[0]['proposed']=='clean box'
    assert results[1]['status']=='review' and results[1]['proposed'] is None
    assert results[1]['validation_error']=='Unsupported noun normalization'


def test_repeated_json_is_accepted_only_for_same_decisions():
    first={'items':[dict(id=0,status='candidate',verb='clean',object_name='box')]}
    text=json.dumps(first)
    assert m.parse_items(text+'\n'+text)==first['items']
    second={'items':[dict(id=0,status='candidate',verb='pack',object_name='box')]}
    with pytest.raises(ValueError,match='Conflicting'):m.parse_items(text+'\n'+json.dumps(second))
