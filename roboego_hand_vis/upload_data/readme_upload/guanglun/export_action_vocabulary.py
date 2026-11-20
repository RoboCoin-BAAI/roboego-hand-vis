"""Export valid action verbs from HumanCOIN_Subtask_Atomic.xlsx."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


def export(source):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    groups = {}
    with zipfile.ZipFile(source) as archive:
        shared = [''.join(node.itertext()) for node in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)] if 'xl/sharedStrings.xml' in archive.namelist() else []
        for number, name, translation in [(1, 'Subtask_100', 'C'), (2, 'Atomic_50', 'K')]:
            entries = []
            for row in ET.fromstring(archive.read(f'xl/worksheets/sheet{number}.xml')).findall('.//m:row', ns):
                cells = {}
                for cell in row.findall('m:c', ns):
                    column = re.sub(r'\d', '', cell.get('r'))
                    if column not in ('A', 'B', translation):
                        continue
                    value = cell.find('m:v', ns)
                    text = value.text if value is not None else ''
                    if cell.get('t') == 's':
                        text = shared[int(text)]
                    elif cell.get('t') == 'inlineStr':
                        text = ''.join(cell.find('m:is', ns).itertext())
                    cells[column] = text.strip()
                verb = cells.get('B', '').lower()
                if re.fullmatch(r'\d+', cells.get('A', '')) and re.fullmatch(r'[a-z]+(?:_[a-z]+)*', verb):
                    entries.append(dict(verb=verb, verb_zh=cells.get(translation, '')))
            groups[name] = entries
    return dict(source=str(source), policy='reference_not_closed_enum', groups=groups,
                counts={key: len(values) for key, values in groups.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.source)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['counts']))
