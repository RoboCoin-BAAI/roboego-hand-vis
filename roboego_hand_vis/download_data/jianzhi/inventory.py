"""Discover Jianzhi session prefixes and export single-column download CSVs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
import subprocess


SOURCE = 'ks3://baai-eai-pwm/jianzhi/'


def discover(source, tool, config, workers=8):
    source = source.rstrip('/') + '/'
    pending, visited, sessions, skipped = [source], set(), set(), set()

    def children(uri):
        result = subprocess.run(
            [str(tool), '-c', str(config), 'ls', uri, '-d', '-L', 'EN', '--loglevel=off'],
            capture_output=True, encoding='utf-8', errors='replace', check=True)
        found = set()
        for line in result.stdout.splitlines():
            path = line.strip()
            if path.startswith(uri) and path != uri and path.endswith('/'):
                relative = path[len(uri):].rstrip('/')
                if '/' not in relative:
                    found.add(path)
        return found

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending:
            current = sorted(set(pending) - visited)
            pending = []
            visited.update(current)
            for parent, paths in zip(current, pool.map(children, current)):
                for path in sorted(paths):
                    name = path.rstrip('/').rsplit('/', 1)[-1]
                    if parent == source and name.startswith('_'):
                        skipped.add(path)
                    elif re.fullmatch(r'[0-9a-fA-F]{32}', name):
                        sessions.add(path)
                    else:
                        pending.append(path)
            print(f'Scanned {len(visited)} directories; {len(sessions)} sessions; '
                  f'{len(pending)} directories pending', flush=True)
    if not sessions:
        raise ValueError('No 32-character session directories found')
    return sorted(sessions), sorted(skipped), len(visited)


def write_batches(output, sessions, batch_size, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob('batch_*.csv')) or (output / 'inventory.json').exists():
        raise ValueError('Output already contains an inventory; use a new output directory')
    batches = {}
    for start in range(0, len(sessions), batch_size):
        name = f'batch_{start // batch_size + 1:07d}.csv'
        rows = sessions[start:start + batch_size]
        with (output / name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['目标路径'])
            writer.writerows([uri] for uri in rows)
        batches[name] = len(rows)
    (output / 'inventory.json').write_text(json.dumps(
        dict(metadata, batch_size=batch_size, total_sessions=len(sessions),
             batches=batches, sessions=sessions), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--ks3-tool', type=Path, required=True)
    parser.add_argument('--ks3-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        parser.error('batch-size and workers must be positive')
    if args.output_dir.exists() and (list(args.output_dir.glob('batch_*.csv')) or
                                   (args.output_dir / 'inventory.json').exists()):
        parser.error('Output already contains an inventory; use a new output directory')
    sessions, skipped, scanned = discover(
        args.source, args.ks3_tool, args.ks3_config, args.workers)
    batches = write_batches(args.output_dir, sessions, args.batch_size,
                            dict(source=args.source, skipped_prefixes=skipped,
                                 scanned_directories=scanned))
    print(f'Saved {len(sessions)} sessions in {len(batches)} CSVs -> {args.output_dir}')


if __name__ == '__main__':
    main()
