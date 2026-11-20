"""Freeze the assistant-reviewed decisions and separate review lists. No remote writes."""
import json
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
data = json.loads((BASE / 'scene_inputs_01_39.json').read_text(encoding='utf-8-sig'))
rows = data['groups']
decisions = json.loads((BASE / 'scene_decisions_draft.json').read_text())
assert not data['errors']
assert len(rows) == len(decisions) == 3464


def assign(ids, category, area, reason):
    for index in ids:
        decisions[index] = dict(id=index, category=category, area=area, reason=reason)


# Explicit corrections after reading the grouped descriptions. These prevent
# incidental nouns (paper money, paper boxes, marker cones, drying racks) from
# deciding a scene.
assign([432, 496, 846, 911, 1143, 1180, 1184, 1202, 1215, 1648, 2512,
        2517, 2558, 2560, 2562, 2564, 2588, 2589, 2620, 2629, 2776,
        2821, 2830, 2951, 3237, 1318], 'other', 'other',
       'Assistant review: game pieces, packaging, generic paper/card sorting or display do not establish a specific scene')
assign([431, 474, 1262, 2466, 2491, 2978], 'household', 'study_room',
       'Assistant review: explicitly domestic desk, lamp or study-stationery organization')
assign([601, 650, 654, 892, 1478, 1495], 'other', 'other',
       'Assistant review: a domestic tabletop-cleaning task does not establish an office or a particular room')
assign([925, 2058], 'household', 'storage_room',
       'Assistant review: domestic storage of personal items in cabinet/drawers')
assign([980], 'household', 'shoe_care_area',
       'Assistant review: domestic shoe cleaning and shoe-cabinet storage')
assign([2497], 'household', 'home_recreation_area',
       'Assistant review: domestic toy-car battery replacement and toy organization')
assign([1943], 'household', 'kitchen',
       'Assistant review: chopstick storage; paper sleeves are packaging, not office documents')
assign([1165, 1173, 1189, 1199, 1200, 1210, 1349, 2548], 'household', 'laundry_area',
       'Assistant review: laundry drying/folding; a clothes-drying rack is not a dish rack')
assign([1039, 1137, 2178], 'household', 'home_fitness_area',
       'Assistant review: explicitly household exercise-equipment care')
assign([249], 'industrial_production', 'workroom',
       'Assistant review: component trimming with pliers at a small work area')
assign([2282], 'commercial_service', 'beverage_shop',
       'Assistant review: takeaway coffee order packaging at a food-service counter')
assign([110, 165, 550, 1824], 'commercial_service', 'food_service_area',
       'Assistant review: drink/tray service or bar-counter cleaning; restaurant versus beverage-shop type is unspecified')
assign([3196], 'commercial_service', 'retail_store',
       'Assistant review: stocking packaged snacks from cartons onto product shelves')
assign([1523], 'industrial_production', 'workroom',
       'Assistant review: explicit product quality inspection at a small work area, no evidence of a factory-scale workshop')

# Review of generic sporting-equipment descriptions: avoid asserting an indoor
# venue when only the equipment-storage function is given.
for row, decision in zip(rows, decisions):
    text = ((row['original_description'] or '') + ' ' + (row['task_description'] or '')).lower()
    if decision['area'] == 'other' or row['id'] in (1681, 1708, 1725, 1733):
        if re.search(r'\b(?:shuttlecocks?|table tennis|badminton equipment|hula hoops?|weightlifting equipment|sports equipment|martial arts belts)\b', text):
            if re.search(r'household|domestic|\bhome\b', text):
                assign([row['id']], 'household', 'home_fitness_area',
                       'Assistant review: explicitly domestic sporting-equipment care/storage')
            else:
                assign([row['id']], 'public_space', 'sports_equipment_area',
                       'Assistant review: sporting-equipment storage/organization, indoor/outdoor setting unspecified')

# User's location knowledge overrides text-only inference (2026-11-18).
user_room_mapping = {
    ('commercial_service', 'massage_room'): ('professional_work', 'conference_center'),
    ('professional_work', 'meeting_room'): ('professional_work', 'conference_center'),
    ('public_space', 'badminton_court'): ('public_space', 'indoor_sports_venue'),
    ('public_space', 'sports_equipment_area'): ('public_space', 'indoor_sports_venue'),
    ('household', 'dressing_area'): ('household', 'bedroom'),
    ('household', 'home_fitness_area'): ('household', 'living_room'),
    ('household', 'home_recreation_area'): ('household', 'living_room'),
    ('household', 'pet_care_area'): ('household', 'living_room'),
    ('household', 'shoe_care_area'): ('household', 'living_room'),
    ('commercial_service', 'food_service_area'): ('commercial_service', 'restaurant'),
}
for decision in decisions:
    pair = decision['category'], decision['area']
    if pair in user_room_mapping:
        category, area = user_room_mapping[pair]
        decision.update(category=category, area=area,
                        reason='User location correction: %s/%s -> %s/%s' % (*pair, category, area))

# Export precisely the user's original Excel taxonomy (columns A-C only).
source = Path('D:/data/annotation_statistics/理想/HumanCOIN_Sence.xlsx')
ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
category_map = {'家庭': 'household', '工业生产': 'industrial_production',
                '商业服务': 'commercial_service', '专业工作': 'professional_work',
                '公共场所': 'public_space', '其他': 'other'}
official = []
with zipfile.ZipFile(source) as archive:
    strings = []
    if 'xl/sharedStrings.xml' in archive.namelist():
        strings = [''.join(node.itertext()) for node in ET.fromstring(
            archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
    for row in ET.fromstring(archive.read('xl/worksheets/sheet1.xml')).findall('.//m:row', ns):
        cells = {}
        for cell in row.findall('m:c', ns):
            column = re.sub(r'\d', '', cell.get('r'))
            if column not in ('A', 'B', 'C'):
                continue
            value = cell.find('m:v', ns)
            text = value.text if value is not None else ''
            if cell.get('t') == 's':
                text = strings[int(text)]
            elif cell.get('t') == 'inlineStr':
                text = ''.join(cell.find('m:is', ns).itertext())
            cells[column] = text
        if cells.get('A') in category_map and cells.get('C'):
            official.append(dict(category=category_map[cells['A']],
                                 category_zh=cells['A'], area=cells['C'], area_zh=cells['B']))
assert len(official) == 38
official_pairs = {(item['category'], item['area']) for item in official}
added_names = {
    ('professional_work', 'meeting_room'): '会议室',
    ('public_space', 'badminton_court'): '羽毛球场地（未确定室内外）',
    ('public_space', 'sports_equipment_area'): '运动器材整理收纳区（未确定室内外）',
}
counts = Counter()
manifest = []
others = []
household_file = BASE / 'scene_household_review.json'
household_overrides = {}
if household_file.exists():
    household_rows = json.loads(household_file.read_text(encoding='utf-8'))
    household_overrides = {item['path']: item for item in household_rows}
    assert len(household_rows) == len(household_overrides) == 950
for row, decision in zip(rows, decisions):
    assert row['id'] == decision['id']
    pair = decision['category'], decision['area']
    assert pair in official_pairs or pair in added_names, pair
    for path in row['paths']:
        item = dict(path=path, category=pair[0], area=pair[1],
                    task_description=row['task_description'],
                    original_description=row['original_description'],
                    reason=decision['reason'])
        if path in household_overrides:
            override = household_overrides[path]
            assert item['task_description'] == override['task_description']
            assert item['original_description'] == override['original_description']
            item.update(category=override['category'], area=override['area'], reason=override['reason'])
        actual_pair = item['category'], item['area']
        assert actual_pair in official_pairs or actual_pair in added_names, actual_pair
        manifest.append(item)
        counts[actual_pair] += 1
        if actual_pair == ('other', 'other'):
            others.append(item)
assert len(manifest) == len({item['path'] for item in manifest}) == 3541
additions = [dict(category=category, area=area, area_zh=added_names[(category, area)],
                  samples=counts[(category, area)])
             for category, area in sorted(counts) if (category, area) not in official_pairs]
output = dict(source_excel=str(source), category_names=category_map,
              official_areas=official, added_areas=additions,
              samples=len(manifest), other_samples=len(others),
              counts=[dict(category=c, area=a, samples=n) for (c, a), n in sorted(counts.items())])
for name, content in [('scene_classification_manifest.json', manifest),
                      ('scene_other_samples.json', others),
                      ('scene_taxonomy_and_counts.json', output)]:
    (BASE / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
with (BASE / 'scene_other_samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['path', 'category', 'area', 'task_description', 'original_description', 'reason'])
    writer.writeheader()
    writer.writerows(others)
print(json.dumps(output, ensure_ascii=False, indent=2))
