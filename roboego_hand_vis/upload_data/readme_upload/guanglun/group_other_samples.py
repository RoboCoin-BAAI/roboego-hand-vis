"""Group unresolved annotations by task content for location review; never edit annotations."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
GROUPS = {
    '01': ('纸箱与包裹处理', '拆箱、折叠压平纸箱、撕标签、整理包裹和包装材料'),
    '02': ('货架商品与库存整理', '瓶罐、食品、清洁用品、药品等在货架或箱篮之间移动整理'),
    '03': ('杯具、杯盖与吸管整理', '堆叠纸杯、收纳杯盖、整理吸管和一次性用品'),
    '04': ('食品饮品准备与分装', '倒水倒饮料、食品分装称重、准备茶点和饮品'),
    '05': ('液体分装与清洁用品补充', '大小瓶之间分装液体、补充洗手液、装泵头和补充清洁剂'),
    '06': ('桌面、柜体与器物清洁', '擦桌面、柜体、玻璃、门把手、水槽和清洗小物件'),
    '07': ('房间地面与垃圾清理', '捡垃圾、扫地吸尘、换垃圾袋、清理地面杂物'),
    '08': ('个人及日用品收纳', '收纳杂物、生活用品、纸巾、眼镜、绳带和小配件'),
    '09': ('文具、纸张、卡片与手工', '纸片卡片分类、纸艺、整理签牌、制作证件挂绳'),
    '10': ('电器、工具维护与整理', '整理线缆、清洁或收纳工具、换电池、维护小电器'),
    '11': ('衣物、布料与洗晒整理', '整理衣物布料、毛巾、围裙、手套和布袋'),
    '12': ('家具陈设与空间布置', '移动桌椅、整理盆栽、拉窗帘、安装架子和调整陈设'),
    '13': ('运动、游戏与玩具用品整理', '准备棋盘桌游、收纳游戏道具和运动用品'),
    '14': ('其他待细分任务', '描述过于笼统或未落入以上任务组的操作'),
}


def classify(row):
    task = (row.get('task_description') or '').lower()
    scene = (row.get('original_description') or '').lower()
    text = task + ' ' + scene
    def has(pattern):
        return bool(re.search(pattern, text))
    # Group by action/object, never by an assumed filming location.
    if has(r'\b(?:chess|monopoly|board game|game box|game components|game props|chessboard|ping pong|table tennis|shuttlecock|badminton|hula hoop|toy|toys|jigsaw|puzzle)\b'):
        return '13'
    if has(r'\b(?:liquid|sanitizer|detergent|soap|powder|granules)\b') and has(r'\b(?:pour\w*|dispens\w*|refill\w*|transfer\w*|distribut\w*|mix\w*|decant\w*|pump|packag\w*|portion\w*)\b'):
        return '05'
    if has(r'\b(?:coffe\w*|tea|teabags?|juice|drink|drinks|hot water|pastries|bread|sandwiches|cand(?:y|ies)|nuts|almonds|peanuts|corn kernels|seeds|hay|sausages|dried fruit|food|foods|snacks)\b') and has(r'\b(?:pour\w*|portion\w*|weigh\w*|packag\w*|seal\w*|scoop\w*|prepar\w*|serv\w*|plating|arrang\w*.*tray)\b'):
        return '04'
    if has(r'\b(?:cleans?|cleaning|wipes?|wiping|washes?|washing|rins\w*|scrub\w*|sanitiz\w*|disinfect\w*)\b'):
        if has(r'\b(?:scissors|tools?|toolbox|tool box|fan grille|computer|humidifier|eyeglasses|eyewear|glasses case|glasses lenses)\b'):
            return '10' if not has(r'\b(?:eyeglasses|eyewear|glasses case|glasses lenses)\b') else '08'
        if has(r'\b(?:vacuum\w*|floor stains|floor edge|floor debris|sweeps?|sweeping|trash can|trash bin|garbage bag|trash bag)\b') and not has(r'\b(?:table|tabletop|countertop|desktop|shelf|shelves|cabinet|door handle|sink)\b'):
            return '07'
        return '06'
    if has(r'\bpaper bags?\b') and has(r'\b(?:fold\w*|seal\w*)\b'):
        return '01'
    if has(r'\b(?:cardboard|cartons?|packages?|packaging materials|paper boxes|gift boxes|packaging boxes|recycling cardboard)\b') and has(r'\b(?:boxes|box|carton|cartons|packages|packaging)\b'):
        if has(r'\b(?:stationery|pens?|pencils?|colored pencils?|markers?|writing instruments)\b') and has(r'\b(?:floor|room)\b'):
            return '07'
        # A carton merely used to unpack goods is not itself the task object.
        if has(r'\b(?:shelf|shelves|shelving)\b') and has(r'\b(?:snacks|bottles|beverages|noodle|food|detergent|shampoo|tissues|toothpaste|medicines?|cotton swab|hand sanitizer|wine|beer|canned|cans)\b'):
            return '02'
        if has(r'\b(?:paper cups?|cup lids?|straws?|cutlery)\b'):
            return '03'
        return '01'
    if has(r'\b(?:trash|garbage|waste|debris|scraps|torn paper|shredded paper|crumpled paper|recycling bins?)\b'):
        return '07' if has(r'\b(?:floor|ground|room|trash bags?|garbage bags?)\b') else '06'
    if has(r'\b(?:anti-slip|chairs?|furniture|curtains?|windows?|potted plants?|flowerpots?|multi-layer storage rack|storage rack.*assembl|shelving units|metal cage|shelf.*install)\b'):
        if not has(r'\b(?:fabric|cloth bags|coiled cables|aprons|wooden sticks|plastic cups|cup lids)\b'):
            return '12'
    if has(r'\b(?:towels?|garments?|clothing|clothes|laundry|fabrics?|cloths|cloth bags|aprons?|socks|shirts?|gloves|shower caps|woven bag|tote bags?|drawstring bags?)\b'):
        return '11'
    if has(r'\b(?:cables?|wires?|earphone|batteries|battery|smoke alarms?|smoke detector|remote controls?|electronic|flashlight|fan base|printing device|handcart|scissors|toolbox|tool box|metal tools?|tape dispenser|tape roll|tape cutter|tape on a dispenser)\b'):
        return '10'
    if has(r'\b(?:paper cups?|cups?|cup lids?|lids|straws?|disposable tableware|cutlery)\b'):
        if has(r'\b(?:wooden sticks|bamboo sticks|black sticks|thin black sticks|white blocks)\b') and not has(r'\bstraws?\b'):
            return '08'
        return '03'
    if has(r'\b(?:shelf|shelves|shelving|stocking|restock\w*|inventory|goods|products)\b'):
        if not has(r'\b(?:rope|belt|straps?|signs|cards|paper strips|sticks|aprons|cloth bags)\b'):
            return '02'
    if has(r'\b(?:cards?|card holders?|ID|id holders?|lanyards?|badges?|paper|origami|glue|labels?|signs?|acrylic|transparent sheets|pen|pens|pen holder|pen refills?|stationery|writing instruments)\b'):
        return '09'
    return '08' if has(r'\b(?:organiz\w*|stor\w*|sort\w*|arrang\w*|tidy\w*|collect\w*|tissues?|personal|accessories|belts?|straps?|ropes?|clips?|sticks?|knots?|whistles?|cabinet|tray mat)\b') else '14'


def main():
    rows = json.loads((BASE / 'scene_other_samples.json').read_text(encoding='utf-8'))
    assert len({row['path'] for row in rows}) == len(rows)
    grouped = {key: [] for key in GROUPS}
    for row in rows:
        key = classify(row)
        grouped[key].append(dict(group_id=key, task_group=GROUPS[key][0], **row))
    flattened = [row for key in GROUPS for row in grouped[key]]
    assert len(flattened) == len(rows)
    fields = ['group_id', 'task_group', 'path', 'task_description', 'original_description', 'category', 'area', 'reason']
    with (BASE / 'scene_other_grouped.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    report = dict(total=len(rows), purpose='Task-content groups for location review; annotations unchanged',
                  groups=[dict(group_id=key, name=GROUPS[key][0], description=GROUPS[key][1],
                               count=len(grouped[key]), samples=grouped[key]) for key in GROUPS])
    (BASE / 'scene_other_grouped.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# other 样本任务分组（供确认场所）', '',
             f'共 {len(rows)} 个样本，按任务内容分组。分组不是最终场景分类，尚未修改服务器标注；不要求同组全部在同一场所。', '',
             '确认场所时可直接回复组号与场景；存在例外的组可以继续细分。', '',
             '| 组号 | 任务大类 | 样本数 | 主要内容 |', '|---|---|---:|---|']
    for key, (name, description) in GROUPS.items():
        if not grouped[key]:
            continue
        lines.append(f'| {key} | {name} | {len(grouped[key])} | {description} |')
    lines += ['', '完整明细见 [scene_other_grouped.csv](scene_other_grouped.csv)（按组号排序，可筛选查看）。', '', '## 各组代表描述', '']
    for key, (name, _) in GROUPS.items():
        if not grouped[key]:
            continue
        lines += [f'### {key} {name}（{len(grouped[key])} 个）', '']
        # Choose common descriptions, followed by a few distinct ones for variety.
        counts = Counter(row['task_description'] for row in grouped[key])
        for task, count in counts.most_common(4):
            lines.append(f'- {task}（同描述 {count} 个）')
        lines.append('')
    (BASE / 'OTHER_TASK_GROUPS.md').write_text('\n'.join(lines), encoding='utf-8')
    for key, (name, _) in GROUPS.items():
        print(key, name, len(grouped[key]))


if __name__ == '__main__':
    main()
