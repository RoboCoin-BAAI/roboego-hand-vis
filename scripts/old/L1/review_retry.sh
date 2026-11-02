#!/usr/bin/env bash
# Annotation-only retry. No VIO setup, conversion, rectification or depth jobs.
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${L1_CONFIG:-$HERE/pipeline.conf}"
: "${DASHSCOPE_API_KEY:?Set the real DASHSCOPE_API_KEY before running}"
export PYTHONUNBUFFERED=1
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$CODE_ROOT/argus${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$STATE_ROOT/review_retry"
exec 9>"$STATE_ROOT/review_retry/.lock"
flock -n 9 || { echo 'Another annotation retry is running' >&2; exit 2; }
if (( $# == 0 )); then
    set -- batch_0000027 batch_0000028 batch_0000029 batch_0000030 \
           batch_0000031 batch_0000032 batch_0000033 batch_0000034
fi
run_dir="$STATE_ROOT/review_retry/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
echo "Retry logs: $run_dir"
child=''
trap 'if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; exit 130' INT TERM
failed=0
for batch in "$@"; do
    [[ "$batch" =~ ^batch_[0-9]{7}$ ]] || { echo "Invalid batch: $batch" >&2; exit 2; }
    selected="$run_dir/$batch.sessions.json"
    # Read the latest original attempt, and retry only its failed video sessions.
    if ! "$PYTHON" - "$STATE_ROOT/$batch" "$OUTPUT_ROOT/$batch" "$selected" <<'PY'
import json
from pathlib import Path
import sys
state_root, output, target = map(Path, sys.argv[1:])
logs = list(state_root.glob('*/review.*.log'))
if not logs:
    raise SystemExit('No original annotation log: ' + str(state_root))
log = max(logs, key=lambda p: p.stat().st_mtime_ns)
state = log.parent
if not any((state / ('review.' + flag)).is_file() for flag in ('done', 'failed')):
    raise SystemExit('Original annotation not finished; skipping: ' + str(state))
allowed = {Path(s).resolve() for s in json.loads((state / 'rectify.sessions.json').read_text())}
outcomes = {}
with log.open(encoding='utf-8', errors='replace') as f:
    for line in f:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get('video') and r.get('status') in ('done', 'failed'):
            session = Path(r['video']).resolve().parents[3]
            if session in allowed and session.is_relative_to(output.resolve()):
                outcomes[str(session)] = r['status']
if not outcomes and allowed:
    raise SystemExit('No per-video outcomes; inspect original log: ' + str(log))
sessions = sorted(s for s, status in outcomes.items() if status == 'failed')
target.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding='utf-8')
target.with_suffix('.original-log.txt').write_text(str(log), encoding='utf-8')
print(f'{output.name}: retry {len(sessions)} failed sessions; original log={log}', flush=True)
PY
    then failed=1; continue; fi
    if [[ $("$PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$selected") == 0 ]]; then
        echo "$batch: no failed sessions"
        continue
    fi
    command=("$PYTHON" "$CODE_ROOT/argus/review/L1/run.py"
        --source "$OUTPUT_ROOT/$batch" --sessions-file "$selected"
        --workers "$REVIEW_WORKERS" --jobs 1 --rig ego_head --no-board
        --model qwen3.8-27b --reasoning none --sample-fps 2 --max-tokens 16384
        --concurrency 1 --cap 0 --estimated-cap-cny "$REVIEW_CAP_CNY")
    printf '%q ' "${command[@]}" > "$run_dir/$batch.command.sh"
    printf '\n' >> "$run_dir/$batch.command.sh"
    echo "[$(date -Is)] Starting annotation retry: $batch"
    setsid "${command[@]}" > "$run_dir/$batch.log" 2>&1 & child=$!
    rc=0; wait "$child" || rc=$?; child=''
    printf '%s\n' "$rc" > "$run_dir/$batch.exitcode"
    # Publish a consolidated latest result for the existing pipeline reporter.
    if ! "$PYTHON" - "$selected" "$run_dir/$batch.log" "$rc" <<'PY'
import json
from pathlib import Path
import sys
from uuid import uuid4
selected, retry_log = map(Path, sys.argv[1:3])
rc = int(sys.argv[3])
original = Path(selected.with_suffix('.original-log.txt').read_text(encoding='utf-8'))
state = original.parent
def outcomes(path):
    result = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get('video') and r.get('status') in ('done', 'failed'):
            result[r['video']] = r
    return result
old, new = outcomes(original), outcomes(retry_log)
expected = set(json.loads(selected.read_text(encoding='utf-8')))
reported = {str(Path(video).resolve().parents[3]) for video in new}
if expected != reported:
    raise SystemExit('Retry outcomes incomplete: original state retained; inspect ' + str(retry_log))
if rc != 0 and all(r['status'] == 'done' for r in new.values()):
    raise SystemExit('Retry exited nonzero despite successful item records; original state retained')
old.update(new)
failed = sum(r['status'] == 'failed' for r in old.values())
log = state / ('review.retry-' + retry_log.parent.name + '.log')
summary = dict(total_videos=len(old), failed=failed)
text = '\n'.join(json.dumps(r, ensure_ascii=False) for r in old.values())
text += '\n' + json.dumps(summary) + '\n'
temp = log.with_name('.' + log.name + '-' + uuid4().hex)
temp.write_text(text, encoding='utf-8')
temp.replace(log)
if failed:
    (state / 'review.done').unlink(missing_ok=True)
    (state / 'review.failed').touch()
    (state / 'review.partial').touch()
else:
    for flag in ('failed', 'partial', 'blocked'):
        (state / ('review.' + flag)).unlink(missing_ok=True)
    (state / 'review.done').touch()
selection = state / 'review.selection.json'
if selection.exists():
    data = json.loads(selection.read_text(encoding='utf-8'))
    data.update(returncode=1 if failed else 0, annotation_total=len(old),
                annotation_succeeded=len(old)-failed, annotation_failed=failed,
                retry_log=str(retry_log))
    temp = selection.with_name('.review-selection-' + uuid4().hex)
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(selection)
print('Original batch annotation status updated: ' + json.dumps(summary), flush=True)
PY
    then failed=1; fi
    echo "[$(date -Is)] $batch exit=$rc log=$run_dir/$batch.log"
    if (( rc != 0 )); then failed=1; fi
done
echo "Annotation retry finished; logs=$run_dir; completed outcomes published to original batch state"
exit "$failed"
