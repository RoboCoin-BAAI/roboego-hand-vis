"""Find VIO candidates from the legacy pipeline ledger, never the dataset tree."""
import json
from pathlib import Path, PurePosixPath


def dataset_root(source, level):
    source = Path(source).resolve()
    return next((p for p in (source, *source.parents) if p.name == f'{level}_lerobot'), source)


def from_pipeline_state(source, level, state_root=None):
    source = Path(source).resolve()
    data_root = dataset_root(source, level)
    state_root = Path(state_root or data_root.parent / f'{level}_pipeline_state')
    if not state_root.is_dir():
        raise ValueError(f'VIO state directory missing: {state_root}; specify --pipeline-state-root or --sessions-file')
    batch_scope = source.name if source.name.startswith('batch_') else (
        source.parent.name if source.parent.name.startswith('batch_') else None)
    batches = [state_root / batch_scope] if batch_scope else sorted(state_root.glob('batch_*'))
    candidates = {}
    for batch in batches:
        print(f'[hand-index] reading {batch}', flush=True)
        logs = []
        for run in sorted(p for p in batch.iterdir() if p.is_dir()) if batch.is_dir() else []:
            files = list(run.glob('vio.*.log'))
            if files:
                logs.append(max(files, key=lambda p:p.stat().st_mtime_ns))
        for log in sorted(logs, key=lambda p:p.stat().st_mtime_ns):
            count = 0
            print(f'[hand-index] VIO log={log}', flush=True)
            with log.open(encoding='utf-8', errors='replace') as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(row, dict) or not isinstance(row.get('episode'), str):
                        continue
                    parts = PurePosixPath(row['episode'].replace('\\','/')).parts
                    if 'data' not in parts:
                        continue
                    # .../SESSION/data/chunk-*/episode_*; use its name under this batch.
                    if len(parts) < 4 or parts[-3] != 'data':
                        continue
                    name = parts[-4]
                    if name in ('.','..'):
                        continue
                    session = data_root / batch.name / name
                    if not session.is_relative_to(source):
                        continue
                    key = (session, parts[-2], parts[-1])
                    candidates[key] = row.get('status') == 'done'
                    count += 1
            print(f'[hand-index] {batch.name}: read {count} episode results', flush=True)
    selected = sorted({key[0] for key, passed in candidates.items() if passed})
    print(f'[hand-index] {len(selected)} candidate sessions from VIO records; current reports will be rechecked', flush=True)
    return selected


def shallow_sessions(source):
    source = Path(source).resolve()
    if (source / 'meta/info.json').is_file():
        return [source]
    children = sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith('.'))
    batches = [p for p in children if p.name.startswith('batch_')]
    sessions = []
    for scope in batches or [source]:
        print(f'[hand-index] listing immediate sessions: {scope}', flush=True)
        sessions.extend(p for p in sorted(scope.iterdir())
                        if not p.name.startswith('.') and (p / 'meta/info.json').is_file())
    return sessions
