"""Download all L3 batches into output_dir/batch_name/session_name.

Run with no arguments on the server to download every L3 batch. Use
--batch-csv selects one batch and uses output_dir/session_name directly,
matching the L1 downloader. Use --dry-run to preview commands.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.parse import urlsplit

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roboego_hand_vis.download_data.L1.run import read_batch


BASE = Path('/share/project/liuyou/humancoin')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--batch-dir', type=Path, default=BASE / 'data/L3_lerobot')
    selection.add_argument('--batch-csv', type=Path, help='Download only this CSV')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'data/L3_lerobot',
                        help='Output root for all batches; exact batch directory with --batch-csv')
    parser.add_argument('--ks3-tool', type=Path,
                        default=BASE / 'utils/ks3/ks3util-linux-amd64')
    parser.add_argument('--ks3-config', type=Path,
                        default=BASE / 'utils/ks3/ks3util-config')
    parser.add_argument('--jobs', type=int, default=64)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')

    batches = ([args.batch_csv.expanduser()] if args.batch_csv else
               sorted(args.batch_dir.expanduser().glob('batch_*.csv')))
    output = args.output_dir.expanduser().resolve()
    tool = args.ks3_tool.expanduser().resolve()
    config = args.ks3_config.expanduser().resolve()
    # Validate every batch before creating directories or downloading anything.
    sessions = {}
    sources_by_name = {}
    try:
        if not batches:
            raise ValueError(f'No batch_*.csv files found in {args.batch_dir}')
        for batch in batches:
            for source in read_batch(batch):
                name = urlsplit(source).path.rstrip('/').rsplit('/', 1)[-1]
                if not name.startswith(('L3-001-', 'L3-002-', 'L3-003-')):
                    raise ValueError(f'Not an L3 session: {source}')
                if name in sources_by_name and sources_by_name[name] != source:
                    raise ValueError(f'Conflicting sources for session {name}')
                sources_by_name[name] = source
                sessions[(batch.stem, name)] = source
        if not args.dry_run:
            if not tool.is_file():
                raise ValueError(f'KS3 tool does not exist: {tool}')
            if not config.is_file():
                raise ValueError(f'KS3 configuration does not exist: {config}')
            if os.name != 'nt' and not os.access(tool, os.X_OK):
                raise ValueError(f'KS3 tool is not executable: {tool}')
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    print(f'{len(batches)} batches, {len(sessions)} session downloads -> {output}', flush=True)
    failed = []
    for index, ((batch_name, name), source) in enumerate(sessions.items(), 1):
        batch_output = output if args.batch_csv else output / batch_name
        destination = batch_output / name
        command = [str(tool), '-c', str(config), '-r', '-u',
                   '--read-timeout', '6', f'--jobs={args.jobs}',
                   'cp', source, str(destination)]
        print(f'[{index}/{len(sessions)}] {source} -> {destination}', flush=True)
        if args.dry_run:
            print(shlex.join(command))
            continue
        destination.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                continue
            reason = f'exit {result.returncode}'
        except OSError as exc:
            reason = str(exc)
        failed.append(source)
        print(f'Download failed ({reason}): {source}', file=sys.stderr, flush=True)

    if args.dry_run:
        print(f'Preview complete: {len(sessions)} sessions; no files downloaded.')
    else:
        print(f'Finished: {len(sessions) - len(failed)} succeeded, {len(failed)} failed.')
        for source in failed:
            print(f'FAILED: {source}', file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

'''
Download all L3 batches (output: L3_lerobot/<batch_name>/<session_name>/):
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/download_data/L3/run.py

Download a single batch (output: L3_lerobot/batch_0000001/<session_name>/):
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/download_data/L3/run.py \
  --batch-csv /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001.csv \
  --output-dir /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001/

Preview all download commands without downloading:
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/download_data/L3/run.py \
  --dry-run
'''

# scp "C:\Users\22563\Desktop\data\L3-lerobot\batch_0000001.csv" "C:\Users\22563\Desktop\data\L3-lerobot\batch_0000003.csv" baai-pwm2-job:/share/project/liuyou/humancoin/data/L3_lerobot/
# scp "C:\Users\22563\Desktop\data\L3-lerobot\*.csv" baai-pwm2-job:/share/project/liuyou/humancoin/data/L3_lerobot/

