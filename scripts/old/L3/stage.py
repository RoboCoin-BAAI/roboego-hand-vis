"""Run a batch stage and publish an explicit allowlist for the next stage."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def select(stage, sessions, records, raw_source=None):
    passed = set()
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get('session'), str):
            continue
        if (stage == 'qc' and item.get('passed') is True
                or stage == 'rectify' and item.get('status') == 'done'):
            passed.add(Path(item['session']).resolve())
    return [s for s in sessions if s in passed]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True)
    p.add_argument('--state', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--raw-source', type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('missing command')
    if a.stage == 'qc':
        with a.input.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.reader(f))[1:]
        names = [urlsplit(row[0].strip()).path.rstrip('/').rsplit('/', 1)[-1]
                 for row in rows if row and row[0].strip()]
        if not names or len(names) != len(set(names)) or any(n in ('', '.', '..') for n in names):
            p.error('invalid or duplicate CSV sessions')
        sessions = [(a.source / n).resolve() for n in names]
        allowlist = a.state / 'qc.input.sessions.json'
        write_json(allowlist, [str(s) for s in sessions])
        command += ['--sessions-file', str(allowlist)]
    else:
        sessions = [Path(s).resolve() for s in json.loads(a.input.read_text(encoding='utf-8'))]
        command += ['--sessions-file', str(a.input)]
    records = []
    rc = 0
    if sessions:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as child:
            for line in child.stdout:
                print(line, end='', flush=True)
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
            rc = child.wait()
    gate = a.stage in ('qc', 'rectify')
    selected = select(a.stage, sessions, records, a.raw_source) if gate else sessions
    # Exit 1 is the entry points' documented per-session failure aggregate.
    # Other errors (signals, argument/import failures) remain stage failures.
    fatal = rc not in (0, 1)
    if a.stage in ('qc', 'rectify') and sessions:
        reported = {Path(r['session']).resolve() for r in records
                    if isinstance(r, dict) and isinstance(r.get('session'), str)}
        fatal = fatal or not set(sessions).issubset(reported)
    skipped = [r for r in records if isinstance(r, dict) and r.get('status') == 'skipped_duration']
    skipped_paths = {r.get('session') for r in skipped}
    excluded = [str(s) for s in sessions if s not in set(selected) and str(s) not in skipped_paths]
    report = dict(stage=a.stage, returncode=rc, total=len(sessions),
                  selected=len(selected), excluded=excluded, skipped_duration=skipped, fatal=fatal)
    write_json(a.state / f'{a.stage}.sessions.json', [str(s) for s in selected])
    write_json(a.state / f'{a.stage}.selection.json', report)
    partial = a.state / f'{a.stage}.partial'
    if excluded or rc:
        partial.touch()
    else:
        partial.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return (rc or 1) if fatal else (0 if gate else rc)


if __name__ == '__main__':
    raise SystemExit(main())
