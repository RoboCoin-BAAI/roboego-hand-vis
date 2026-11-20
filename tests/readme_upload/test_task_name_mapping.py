import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[2]/'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0,str(BASE))
import build_task_name_mapping as m

RULES=json.loads((BASE/'task_name_normalization_rules.json').read_text())
VERBS={'wipe','clean','organize','sort','stack','pack','seal','wash','arrange','cut','trim','wear','polish'}


def test_objects_before_verbs():
    r=m.normalize(' Wipe   Boxes ',RULES,VERBS)
    assert r['object_stage']=='wipe box' and r['proposed']=='clean box'
    assert r['rules']==['format','object_plural','verb_alias:wipe->clean']


def test_no_blind_s_suffix_or_compound_truncation():
    for noun in ('glass','glasses','scissors','pants','cosmetics','lip gloss'):
        assert m.normalize('clean '+noun,RULES,VERBS)['proposed']=='clean '+noun
    r=m.normalize('wipe counter and mirror',RULES,VERBS)
    assert r['status']=='review_compound' and r['proposed']=='wipe counter and mirror'
    assert m.normalize('wipe hair with towel',RULES,VERBS)['status']=='review_compound'


def test_task_targets_and_unapproved_verbs_are_preserved():
    for task in ('sort box','stack box','pack box','seal box','wash hair','trim hair','polish table','arrange box'):
        assert m.normalize(task,RULES,VERBS)['proposed']==task
    assert m.normalize('wipe mirrors',RULES,VERBS)['proposed']=='clean mirror'
    assert m.normalize('organize cartons',RULES,VERBS)['proposed']=='organize carton'
    assert m.normalize('organize clothes',RULES,VERBS)['proposed']=='organize clothing'
