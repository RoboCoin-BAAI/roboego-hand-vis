"""Upload completed depth sessions to their original KS3 prefixes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roboego_hand_vis.upload_data.ks3 import prefix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--ks3-tool', type=Path, required=True)
    parser.add_argument('--ks3-config', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=64)
    parser.add_argument('--read-timeout', type=int, default=6)
    parser.add_argument('--first-batch', type=int, default=1)
    parser.add_argument('--last-batch', type=int, default=9999999)
    parser.add_argument('--execute', action='store_true', help='Otherwise only print source and destination')
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.read_timeout < 1 or not 1 <= args.first_batch <= args.last_batch:
        parser.error('Invalid jobs, timeout or batch range')
    if not args.state_root.is_dir():
        parser.error('state-root does not exist')
    for path in (args.ks3_tool, args.ks3_config):
        if not path.is_file():
            parser.error('Missing file: ' + str(path))

    total, success, failures = 0, 0, []
    for batch in sorted(args.state_root.glob('batch_*')):
        if not batch.name[6:].isdigit() or not args.first_batch <= int(batch.name[6:]) <= args.last_batch:
            continue
        done_file = batch / 'depth_done.json'
        if not done_file.is_file():
            continue
        done = set(json.loads(done_file.read_text(encoding='utf-8')))
        for record_file in sorted((batch / 'depth_records').glob('*.json')):
            try:
                record = json.loads(record_file.read_text(encoding='utf-8'))
                if record.get('status') != 'done' or record.get('source_uri') not in done:
                    continue
                total += 1
                source = Path(record['session'])
                destination = prefix(record['source_uri'])
                if not source.is_dir() or not any(p.is_file() for p in (source / 'depth').rglob('*')):
                    raise ValueError('Session or depth results missing: ' + str(source))
                print(f'UPLOAD {source}/ -> {destination}', flush=True)
                if args.execute:
                    subprocess.run([
                        str(args.ks3_tool), '-c', str(args.ks3_config), '-r', '-u',
                        '--read-timeout', str(args.read_timeout), '--jobs=' + str(args.jobs),
                        'cp', str(source) + '/', destination,
                    ], check=True)
                    success += 1
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
                failures.append(str(record_file))
                print(f'FAILED {record_file}: {exc}', flush=True)
    print(f'Sessions: {total}; uploaded: {success}; failed: {len(failures)}; execute: {args.execute}', flush=True)
    for path in failures:
        print('FAILED: ' + path, flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
