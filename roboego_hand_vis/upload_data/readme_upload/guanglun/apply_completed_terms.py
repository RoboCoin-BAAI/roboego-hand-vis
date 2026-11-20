"""Apply a frozen batch range from running term-cleaning progress, without API calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from types import SimpleNamespace

try:
    from . import clean_names_and_actions_qwen as combined
except ImportError:
    import clean_names_and_actions_qwen as combined

BASE = Path(__file__).resolve().parent


def snapshot(progress, start, end):
    selected = {}
    lines = progress.read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue  # A running producer may be appending its final line.
            raise
        match = re.fullmatch(r'batch_(\d+)/[^/]+/annotation/episode_\d+_structured\.json', row['path'])
        if not match:
            raise ValueError('Unsafe progress path: ' + row['path'])
        if start <= int(match[1]) <= end:
            if row['path'] in selected:
                raise ValueError('Duplicate progress path: ' + row['path'])
            selected[row['path']] = row
    return [selected[key] for key in sorted(selected)]


class NoAPI:
    @staticmethod
    def settings(*args):
        raise RuntimeError('Cache-only recovery: API dispatch is forbidden')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=35)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--model', default='qwen3.7-plus')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--recover-return-cache', action='store_true')
    args = parser.parse_args(argv)
    args.root, args.state_dir, args.report_dir = (p.resolve() for p in (args.root, args.state_dir, args.report_dir))
    if not 1 <= args.start <= args.end or args.workers < 1:
        parser.error('Invalid batch range or worker count')
    if args.report_dir == args.root or args.root in args.report_dir.parents:
        parser.error('Reports and backups must be outside the dataset')
    if args.report_dir.exists():
        parser.error('Use a fresh report directory to preserve earlier backups/reports')
    rows = snapshot(args.state_dir / 'prepare_progress.jsonl', args.start, args.end)
    if not rows:
        parser.error('No completed/error progress records in selected range')
    args.report_dir.mkdir(parents=True)
    combined.save_json(args.report_dir / 'source_snapshot.json', rows)
    recovery = []
    if args.recover_return_cache:
        object_words = {e['name'] for e in json.loads((BASE/'object_vocabulary.json').read_text(encoding='utf-8'))['entries']}
        action_words = {key: [e['verb'] for e in values] for key, values in
                        json.loads((BASE/'action_vocabulary.json').read_text(encoding='utf-8'))['groups'].items()}
        cfg = SimpleNamespace(root=args.root, state_dir=args.state_dir, model=args.model,
                              max_tokens=args.max_tokens, timeout=180, retry_failed=False,
                              apply=False, dry_run=False)
        for i, row in enumerate(rows):
            if row['status'] != 'error' or not row.get('error', '').endswith('repeated-execution label: return'):
                continue
            try:
                recovered = combined.prepare_one(args.root / row['path'], cfg, NoAPI, object_words, action_words)
                rows[i] = recovered
                recovery.append(dict(path=row['path'], status='done'))
            except Exception as exc:
                recovery.append(dict(path=row['path'], status='error', error=str(exc)))
                rows[i] = dict(row, original_error=row.get('error'), error=str(exc))
        combined.save_json(args.report_dir / 'cache_recovery.json', recovery)
    # Independent immutable input; the running model job's bundle/progress is untouched.
    combined.save_json(args.report_dir / 'selected_bundle.json', dict(version=1, status='complete',
         root=str(args.root), scope=['objects[].name', 'actions.vocabulary'], records=rows))
    valid = [row for row in rows if row['status'] == 'done']
    pending = [row for row in rows if row['status'] != 'done']
    combined.save_json(args.report_dir / 'pending_preparation.json', pending)
    results = []
    with (args.report_dir / 'apply_progress.jsonl').open('w', encoding='utf-8') as log, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(combined.apply_record, args.root, row, args.report_dir): row for row in valid}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = dict(path=futures[future]['path'], status='error', error=str(exc))
            results.append(result)
            log.write(json.dumps(result, ensure_ascii=False) + '\n'); log.flush()
            if len(results) % 100 == 0 or result['status'] == 'error':
                print(json.dumps(dict(completed=len(results), total=len(valid), last=result)), flush=True)
    summary = dict(selected=len(rows), valid=len(valid), pending_preparation=len(pending),
                   recovered_from_cache=sum(r['status']=='done' for r in recovery),
                   changed=sum(r.get('changed',False) for r in results),
                   unchanged=sum(r['status']=='done' and not r['changed'] for r in results),
                   write_errors=sum(r['status']=='error' for r in results))
    combined.save_json(args.report_dir / 'apply_report.json', dict(root=str(args.root), start=args.start,
                        end=args.end, summary=summary, records=results))
    print(json.dumps(summary), flush=True)
    return int(summary['write_errors'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
