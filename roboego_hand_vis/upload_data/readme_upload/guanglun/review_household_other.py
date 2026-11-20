"""Review the original 950 unresolved samples using the user's household prior."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
AREA_ZH = {'kitchen': '厨房', 'living_room': '客厅', 'dining_room': '家庭餐厅',
           'bedroom': '卧室', 'study_room': '书房', 'bathroom': '卫生间',
           'storage_room': '储物间', 'laundry_area': '洗衣区', 'balcony': '阳台', 'other': '待确认'}


def classify(row):
    text = ((row.get('task_description') or '') + ' ' + (row.get('original_description') or '')).lower()
    task = (row.get('task_description') or '').lower()
    group = row['group_id']
    def has(pattern):
        return bool(re.search(pattern, text))
    def result(area, basis):
        return area, 'User requested household context; inferred room: ' + basis
    if group == '14':
        return result('other', 'bottles/base assembly does not identify a household room')
    if group == '13':
        return result('living_room', 'home games, toy cleaning and sports-item care; user confirmed living room')
    if has(r'\b(?:balcony|laundry|washing machine|clothesline)\b'):
        return result('balcony' if 'balcony' in text else 'laundry_area', 'explicit balcony or laundry context')
    if has(r'\b(?:toiletries|mouthwash|contact lens|soap dispenser|hand soap|hand sanitizer|shower caps|toothbrush|toothpaste|bathroom)\b') and group not in ('01', '02'):
        return result('bathroom', 'household hygiene supplies or washing-up care')
    if has(r'\b(?:jewelry|jewellery|makeup|cosmetics|dressing table|necklace|bracelet)\b'):
        return result('bedroom', 'personal grooming/accessories; user confirmed this activity in bedroom')
    if group == '01':
        return result('storage_room', 'household carton/package storage, unpacking or recycling preparation')
    if group == '02':
        if has(r'\b(?:kitchenware|spice|stainless steel basin|food containers|refrigerator|fridge)\b'):
            return result('kitchen', 'kitchen vessel, spice or refrigerator organization')
        return result('storage_room', 'household shelves, stored supplies and stock organization')
    if group == '03':
        return result('kitchen', 'organization of cups, lids, straws and tableware')
    if group == '04':
        if has(r'\b(?:host\w*|guests|hospitality|refreshments)\b'):
            return result('living_room', 'household guest refreshments and hospitality')
        return result('kitchen', 'household food/drink preparation and portioning')
    if group == '05':
        if has(r'aromatherapy|diffuser'):
            return result('living_room', 'household fragrance refill')
        if has(r'\b(?:detergent|soap|sanitizer|cleaner)\b'):
            return result('bathroom', 'identified personal-hygiene or household cleaning-liquid refill')
        return result('other', 'unspecified liquid, powder or granules cannot distinguish kitchen/bathroom/other room')
    if group == '06':
        if has(r'\b(?:sink|basin|faucet|washstand|mirror|comb)\b'):
            if has(r'\b(?:ashtray|cups?|glassware|utensils|dish|dishes|tray|drying rack)\b'):
                return result('kitchen', 'washing vessels or small household items at a sink')
            return result('bathroom', 'household washstand, sink or mirror cleaning')
        if has(r'\b(?:ashtray|washes? .*cups?|rins\w* .*cups?|washing.*drinking)\b'):
            return result('kitchen', 'washing household cups or ashtrays')
        return result('living_room', 'general household tables, cabinets, glass, handles and surface cleaning')
    if group == '07':
        return result('living_room', 'general household room/floor tidying and waste cleanup')
    if group == '08':
        if has(r'\b(?:books?|booklets|bookshelf|pens?|pencils?|stationery|schoolbag)\b'):
            return result('study_room', 'household reading or study-material organization')
        if has(r'\b(?:tissues?|glasses|eyeglasses|eyewear|paper cup|water bottles?|umbrellas?|jump ropes?|belts?|straps?|wristbands?|whistles?|ropes?|clips?|hygiene)\b'):
            return result('living_room', 'daily personal items and sports accessories; user located home sporting-item care in living room')
        return result('storage_room', 'general household small-item and storage-container organization')
    if group == '09':
        return result('study_room', 'household paper/card/label sorting, stationery or crafts')
    if group == '10':
        if has(r'\b(?:smoke alarm|smoke alarms|smoke detector|smoke detectors)\b'):
            return result('other', 'smoke-alarm battery maintenance does not establish which room')
        if has(r'\b(?:data cable|earphone|phone|power bank|printing device|tape dispenser|tape roll|tape cutter|tape on a dispenser)\b'):
            return result('study_room', 'household charging accessories, printing or stationery equipment')
        if has(r'\b(?:remote control|fan|humidifier)\b'):
            return result('living_room', 'common household electronics care or remote-control battery replacement')
        return result('storage_room', 'household tool, cable-reel or utility-equipment maintenance/storage')
    if group == '11':
        if has(r'\b(?:aprons?)\b'):
            return result('kitchen', 'household cooking-apron storage')
        if has(r'\b(?:shirt|shirts|t-shirt|garments?|clothing|clothes|towels?|fabric.*trim|trims?.*fabric)\b'):
            return result('laundry_area', 'household clothing/linen handling or fabric-edge trimming')
        if has(r'\b(?:wristband|air pump|non-slip gloves|black gloves)\b'):
            return result('living_room', 'home sport/exercise accessories, consistent with user location correction')
        return result('storage_room', 'household bags, fabric pieces and gloves stored on racks/in containers')
    if group == '12':
        if has(r'\b(?:storage rack|shelving|shelf division|partitions)\b'):
            return result('storage_room', 'household storage-rack installation or shelf partitioning')
        return result('living_room', 'common-room furniture, plants, windows, carpet and household layout')
    if group == '13':
        return result('living_room', 'home chess/board games and sports/toy organization; user confirmed living room')
    raise ValueError('Unknown task group: ' + group)


def main():
    source = BASE / 'scene_other_grouped_initial950.json'
    if not source.exists():
        source.write_bytes((BASE / 'scene_other_grouped.json').read_bytes())
    groups = json.loads(source.read_text(encoding='utf-8'))['groups']
    output = []
    for group in groups:
        for row in group['samples']:
            area, reason = classify(row)
            output.append(dict(path=row['path'], group_id=row['group_id'], task_group=row['task_group'],
                               category='other' if area == 'other' else 'household', area=area,
                               task_description=row['task_description'], original_description=row['original_description'], reason=reason))
    assert len(output) == len({row['path'] for row in output}) == 950
    (BASE / 'scene_household_review.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (BASE / 'scene_household_review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    counts = Counter(row['area'] for row in output)
    lines = ['# 原 950 个 other 样本的家庭场景复分', '',
             '根据用户“先尝试归类到家庭中”的要求，以家庭为前提，依据任务和原场景描述推断房间；不是已核实的拍摄房间。原描述原文不变。', '',
             '| 房间 | area | 数量 |', '|---|---|---:|']
    for area, count in counts.most_common():
        lines.append(f'| {AREA_ZH[area]} | `{area}` | {count} |')
    lines += ['', '## 原任务组的去向', '', '| 组号 | 任务大类 | 分配结果 |', '|---|---|---|']
    for group in groups:
        group_counts = Counter(row['area'] for row in output if row['group_id'] == group['group_id'])
        summary = '；'.join(f'{AREA_ZH[area]} {n}' for area, n in group_counts.most_common())
        lines.append(f"| {group['group_id']} | {group['name']} | {summary} |")
    lines += ['', '完整复分结果见 [scene_household_review.csv](scene_household_review.csv)。原 950 个样本的任务分组快照保留在 `scene_other_grouped_initial950.json`。', '',
              '仍为 other 的主要原因：液体/粉末/颗粒内容未说明、烟雾报警器操作无法判断房间、瓶子与底座装配缺少场所依据。', '']
    (BASE / 'HOUSEHOLD_OTHER_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    print(dict(counts))


if __name__ == '__main__':
    main()
