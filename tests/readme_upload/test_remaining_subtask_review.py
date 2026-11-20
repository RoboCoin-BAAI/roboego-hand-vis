import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[2]/'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0,str(BASE))
import review_remaining_by_subtasks as m
RULES=json.loads((BASE/'task_name_normalization_rules.json').read_text())


def segment(text,start=0,end=30):return dict(text=text,start_s=start,end_s=end)


def test_noun_verb_spellings_do_not_invent_actions():
    events=m.parse_segment(segment('Place the nail polish on the shelf'),['nail polish','shelf'],RULES,'Organize products on shelf')
    assert [(x['verb'],x['noun']) for x in events]==[('organize','nail polish')]
    events=m.parse_segment(segment('Put the baking tray into the oven'),['tray','oven'],RULES,'Bake bread')
    assert all(x['verb']!='bake' for x in events)


def test_pronoun_and_cleaning_surface_have_explicit_antecedents():
    events=m.parse_segment(segment('Pick up the cup and rinse it in the sink'),['cup','sink'],RULES,'Wash cup')
    assert [(x['verb'],x['noun']) for x in events]==[('rinse','cup')]
    events=m.parse_segment(segment('Wipe the stains on the table with a cloth'),['table','cloth'],RULES,'Clean table')
    assert [(x['verb'],x['noun']) for x in events]==[('clean','table')]


def test_ties_do_not_force_a_primary_object_and_invalid_times_are_flagged():
    base=dict(dataset='test',path='p',sha256='s',old_task_name='Clean table and mirror',task_description='',
              objects=[{'name':'table'},{'name':'mirror'}],subtasks=[segment('Wipe the table',0,20),segment('Wipe the mirror',20,40)])
    result=m.review(base,RULES)
    assert result['status']=='review' and result['top_to_second']==1
    base['subtasks']=[segment('Wipe the table',20,10)]
    assert 'Invalid' in m.review(base,RULES)['reason']


def test_back_is_a_body_target_not_a_directional_stopword():
    events=m.parse_segment(segment("Massage the customer's back"),['back'],RULES,'Massage back')
    assert events[0]['noun']=='back'
