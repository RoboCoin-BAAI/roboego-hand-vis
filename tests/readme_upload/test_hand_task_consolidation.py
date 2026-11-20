import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[2]/'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0,str(BASE))
import consolidate_hand_task_mapping as m
RULES=json.loads((BASE/'task_name_normalization_rules.json').read_text())


def test_storage_aliases_keep_different_physical_goals():
    assert m.normalize_pair('arrange','boxes',RULES)[:2]==('organize','box')
    assert m.normalize_pair('wipe','counter',RULES)[:2]==('clean','counter')
    for verb in ['sort','stack','pack','seal','wash','polish']:
        assert m.normalize_pair(verb,'box',RULES)[:2]==(verb,'box')
    assert m.normalize_pair('arrange','hair',RULES)[:2]==('arrange','hair')


def test_body_treatment_is_not_object_scrubbing():
    assert m.normalize_pair('scrub','face',RULES)[:2]==('scrub','face')
    assert m.normalize_pair('scrub','table',RULES)[:2]==('clean','table')
    assert m.normalize_pair('massage','customer legs',RULES)[:2]==('massage','leg')


def test_service_wrappers_and_auxiliary_tools():
    assert m.phrase_rules('Perform facial massage')[:2]==('massage','face')
    assert m.phrase_rules('Inflate basketball with air pump')[:2]==('inflate','basketball')
    assert m.phrase_rules('Massage customer back and head') is None
    assert m.phrase_rules('Perform facial care') is None
    assert not m.valid_pair('clean','bathroom',{'clean'})
