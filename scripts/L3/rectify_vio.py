"""Run L3 stereo rectification then stereo VIO for an explicit JSON session list."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


REPO = Path(__file__).resolve().parents[2]


def run_stage(command, log_path, label):
    print(f'[{label}] start; log={log_path}', flush=True)
    records = []
    with log_path.open('w', encoding='utf-8') as log:
        log.write(json.dumps(dict(command=command), ensure_ascii=False) + '\n')
        log.flush()
        child = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                 errors='replace', bufsize=1)
        try:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(f'[{label}] {line.rstrip()}', flush=True)
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    records.append(row)
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
        finally:
            child.stdout.close()
    return dict(returncode=code, log=str(log_path), records=records)


def process(session, index, args, output):
    folder = output / f'{index:04d}_{session.name}'
    folder.mkdir()
    result = dict(session=str(session), status='failed', stages={})
    try:
        if not session.is_dir() or not (session / 'meta/info.json').is_file():
            raise ValueError(f'Not a converted LeRobot session: {session}')
        commands = [
            ('rectify', [str(args.python), '-u', '-m', 'roboego_hand_vis.lerobot_stereo_rectification.L3.run',
                         '--source', str(session), '--workers', '1', '--ffmpeg', args.ffmpeg,
                         '--balance', str(args.balance)]),
            ('vio', [str(args.python), '-u', '-m', 'roboego_hand_vis.lerobot_vio.L3.run',
                     '--source', str(session), '--workers', '1', '--openvins-root', str(args.openvins_root),
                     '--input-mode', 'rectified', '--time-offset-mode', args.time_offset_mode,
                     '--timing-bootstrap', args.timing_bootstrap, '--init-mode', args.init_mode]),
        ]
        if args.force_rectify:
            commands[0][1].append('--no-skip-existing')
        if args.zupt:
            commands[1][1].append('--zupt')
        elif args.zupt_fallback:
            commands[1][1].append('--zupt-fallback')
        for stage, command in commands:
            result['stages'][stage] = run_stage(command, folder / f'{stage}.log', f'{index}/{session.name}/{stage}')
            if result['stages'][stage]['returncode'] != 0:
                result['status'] = stage + '_failed'
                break
        else:
            result['status'] = 'done'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(session=str(session), status=result['status'], logs=str(folder)), ensure_ascii=False), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sessions-file', required=True, type=Path)
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--openvins-root', type=Path, default=REPO.parent / 'open_vins')
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--workers', type=int, default=1, help='Concurrent sessions; each stage uses one worker')
    p.add_argument('--log-dir', type=Path, help='Parent of a new run directory')
    p.add_argument('--balance', type=float, default=0.)
    p.add_argument('--force-rectify', action='store_true')
    p.add_argument('--time-offset-mode', choices=('estimate', 'config'), default='estimate')
    p.add_argument('--init-mode', choices=('static-zupt', 'dynamic', 'static-then-dynamic', 'dynamic-then-static', 'auto', 'static'), default=None)
    p.add_argument('--timing-bootstrap', choices=('auto', 'off'), default='auto')
    policy = p.add_mutually_exclusive_group()
    policy.add_argument('--zupt', action='store_true')
    policy.add_argument('--zupt-fallback', action='store_true')
    args = p.parse_args(argv)
    if args.init_mode is None:
        args.init_mode = 'auto' if args.zupt or args.zupt_fallback else 'static-then-dynamic'
    if args.init_mode in ('static-zupt', 'dynamic', 'static-then-dynamic', 'dynamic-then-static') and (args.zupt or args.zupt_fallback):
        p.error('Use --init-mode alone; do not combine with legacy --zupt/--zupt-fallback')
    if args.workers < 1 or not 0 <= args.balance <= 1:
        p.error('workers must be positive; balance must be in [0, 1]')
    values = json.loads(args.sessions_file.read_text(encoding='utf-8-sig'))
    if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
        p.error('sessions-file must be a nonempty JSON list of session paths')
    if any(not Path(v).is_absolute() for v in values):
        p.error('Session paths must be absolute')
    sessions = list(dict.fromkeys(Path(v).resolve() for v in values))
    args.python = args.python.expanduser().resolve()
    args.openvins_root = args.openvins_root.expanduser().resolve()
    parent = args.log_dir or args.sessions_file.resolve().parent / 'rectify_vio_runs'
    output = parent / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'sessions.json').write_text(json.dumps([str(s) for s in sessions], indent=2), encoding='utf-8')
    print(f'Sessions={len(sessions)}; concurrent={args.workers}; logs={output}', flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # JSONL is appended on the main thread, preserving partial results if interrupted.
        futures = [pool.submit(process, session, index, args, output)
                   for index, session in enumerate(sessions, 1)]
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with (output / 'results.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
    counts = dict(Counter(r['status'] for r in results))
    summary = dict(total=len(sessions), counts=counts, log_dir=str(output))
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return int(any(r['status'] != 'done' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
