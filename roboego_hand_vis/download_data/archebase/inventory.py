"""Export Archebase episode directories as Jianzhi-style download CSVs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from roboego_hand_vis.download_data.jianzhi.inventory import write_batches


SOURCE = 'ks3://baai-eai-pwm/archebase_1000/'


def discover(source, tool, config):
    source = source.rstrip('/') + '/'
    result = subprocess.run(
        [str(tool), '-c', str(config), 'ls', source, '-L', 'EN', '--loglevel=off'],
        capture_output=True, encoding='utf-8', errors='strict', check=True)
    files = {}
    for line in result.stdout.splitlines():
        # Preserve spaces in scene/task names and allow timestamps with a zone.
        offset = line.find(source)
        if offset < 0:
            continue
        fields = line[:offset].split()
        if len(fields) < 3:
            continue
        uri = line[offset:].rstrip()
        parent, name = uri.rsplit('/', 1)
        if name in ('data.mcap', 'annotation.json'):
            if int(fields[-3]) <= 0:
                raise ValueError('Empty episode file: ' + uri)
            files.setdefault(parent + '/', set()).add(name)
    if not files:
        raise ValueError('No episode files found')
    incomplete = {uri: sorted(names) for uri, names in files.items()
                  if names != {'data.mcap', 'annotation.json'}}
    if incomplete:
        raise ValueError('Incomplete episodes: ' + json.dumps(incomplete, ensure_ascii=False))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--ks3-tool', type=Path, required=True)
    parser.add_argument('--ks3-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if (list(args.output_dir.glob('batch_*.csv')) or
            (args.output_dir / 'inventory.json').exists()):
        parser.error('Output already contains an inventory; use a new output directory')
    sessions = discover(args.source, args.ks3_tool, args.ks3_config)
    source = args.source.rstrip('/') + '/'
    scenes = Counter(uri[len(source):].split('/')[0] for uri in sessions)
    batches = write_batches(args.output_dir, sessions, args.batch_size,
                            dict(source=source, scenes=dict(sorted(scenes.items())),
                                 required_files=['data.mcap', 'annotation.json']))
    print(f'Saved {len(sessions)} episodes in {len(batches)} CSVs -> {args.output_dir}')


if __name__ == '__main__':
    main()
