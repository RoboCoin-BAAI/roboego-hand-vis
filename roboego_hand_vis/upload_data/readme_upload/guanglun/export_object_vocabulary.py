"""Export reference noun names from HumanCOIN_Object.xlsx (standard library only)."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


def export(source):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    entries = []
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
                cells[column] = text.strip()
            # Exclude header and trailing prose such as "639 个标准物品词".
            if not re.fullmatch(r'\d+', cells.get('A', '')) or not cells.get('B'):
                continue
            entries.append(dict(name=cells['B'].lower(), name_zh=cells.get('C', '')))
    if not entries or len(entries) != len({row['name'] for row in entries}):
        raise ValueError('Empty or duplicated object vocabulary')
    return dict(source=str(source), policy='reference_not_closed_enum', count=len(entries), entries=entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.source)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), count=result['count'])))
