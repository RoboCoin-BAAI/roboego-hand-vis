import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[2]/'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0,str(BASE))
import build_common_tasks_31 as m
T=json.loads((BASE/'OBJECT_CATEGORIES_31.json').read_text(encoding='utf-8'))


def test_approved_taxonomy_has_exactly_31_classes():
    assert len(T['categories'])==31
    assert m.classify('chair','clean','',T)[0]=='furniture'
    assert m.classify('face','massage','',T)[0]=='body'


def test_material_and_container_remain_distinct():
    assert m.classify('bottle','clean','lotion bottle',T)[0]=='container'
    assert m.classify('lotion','apply','apply lotion',T)[0]=='cosmetics'
    assert m.classify('cup','wash','',T)[0]=='tableware'
    assert m.classify('hair dye','apply','',T)[0]=='hair_dye'
    assert m.classify('nail polish','apply','',T)[0]=='nail_polish'


def test_contextual_ambiguity_not_a_blind_keyword_assignment():
    assert m.classify('dye','apply','dye the hair',T)[0]=='hair_dye'
    assert m.classify('dye','apply','dye fabric',T)[0] is None
    assert m.classify('brush','organize','hairdresser perm tools',T)[0]=='beauty_tool'
    assert m.classify('brush','organize','cleaning brush for the floor',T)[0]=='cleaning_tool'
    assert m.classify('glass','clean','window glass panel',T)[0]=='glass'


def test_specific_tools_and_catalog_gaps_are_protected():
    assert m.classify('nail drill','clean','',T)[0]=='beauty_tool'
    assert m.classify('hair dryer','clean','',T)[0]=='appliance'
    assert m.classify('book','organize','',T)[0]=='paper_material'
    assert m.classify('folder','organize','',T)[0]=='stationery'
    assert m.classify('floor','clean','',T)[0]=='floor'
    assert m.classify('surface','clean','',T)[0]=='item'


def test_user_added_members_and_independent_categories():
    for noun,target in [('shoelace','clothing'),('straw','tableware'),('ball','sports_equipment'),('sausage','food')]:
        assert m.classify(noun,'organize','',T)[0]==target
    for noun in ['floor','sink','mirror','door']:
        assert m.classify(noun,'clean','',T)[0]==noun


def test_liquid_and_glass_follow_description_not_container_alone():
    assert m.classify('liquid','pour','refill the soap dispenser',T)[0]=='toiletries'
    assert m.classify('liquid','apply','apply to nail tips and cure in UV lamp',T)[0]=='nail_polish'
    assert m.classify('liquid','pour','pour liquid between unspecified bottles',T)[0]=='liquid'
    assert m.classify('glass','clean','wipe the glass display cabinet',T)[0]=='furniture'
    assert m.classify('glass','rinse','wash glasses in a kitchen sink',T)[0]=='tableware'


def test_generic_noun_families_merge_but_specific_categories_win():
    for noun in ['product','object','item','cylindrical object','long item','household item','office supply','raw material','kitchen item']:
        assert m.classify(noun,'organize','',T)[0]=='item'
    assert m.classify('skin care product','organize','',T)[0]=='cosmetics'
    assert m.classify('medicinal material','organize','',T)[0]=='medicine'
