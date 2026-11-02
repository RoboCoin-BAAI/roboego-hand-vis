"""Read-only TXT audit of L1 pipeline state and per-session failures (stdlib only)."""
import argparse
from collections import Counter, deque
from datetime import datetime
import json
from pathlib import Path

STAGES = ('download', 'convert', 'qc', 'rectify', 'depth', 'vio', 'review')


def load(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def records(path):
    result, tail = [], deque(maxlen=12)
    if path is not None:
        with path.open(encoding='utf-8', errors='replace') as f:
            for line in f:
                tail.append(line.rstrip())
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        result.append(item)
                except ValueError:
                    pass
    return result, list(tail)


def latest(paths):
    return max(paths, key=lambda p: p.stat().st_mtime_ns, default=None)


def stage_status(state, stage):
    if (state / f'{stage}.failed').exists():
        return 'FAILED'
    if (state / f'{stage}.blocked').exists():
        return 'BLOCKED'
    if (state / f'{stage}.done').exists():
        return 'PARTIAL' if (state / f'{stage}.partial').exists() else 'DONE'
    return 'STARTED' if list(state.glob(f'{stage}.*.log')) else '-'


def audit(root, output_root, out):
    def emit(value=''):
        print(value, file=out)
    emit('L1 流水线详细报告  ' + datetime.now().astimezone().isoformat(timespec='seconds'))
    emit(f'状态根目录：{root}\n数据根目录：{output_root}')
    emit('每批选择最近修改的运行目录；只分析各阶段最后一次日志，不混算重试。')
    emit('DONE 仅表示阶段结束，空输入也会 DONE；VIO/depth/review 的 selected 是输入数，不是成功数。')
    emit('STARTED 不证明进程仍存活；产物/子日志可能被后续运行改写，优先以阶段日志为依据。')
    totals = Counter()
    batches = []
    for batch in sorted(root.glob('batch_*')):
        candidates = [p for p in batch.iterdir() if p.is_dir()]
        state = max(candidates, key=lambda p: max((f.stat().st_mtime_ns for f in p.iterdir() if f.is_file()), default=0), default=None)
        if state:
            batches.append((batch, state))
    emit('\nbatch              ' + ' '.join(f'{s:>10}' for s in STAGES))
    for batch, state in batches:
        statuses = [stage_status(state, s) for s in STAGES]
        emit(f'{batch.name:<19}' + ' '.join(f'{s:>10}' for s in statuses))
        for stage, status in zip(STAGES, statuses):
            totals[f'{stage}/{status}'] += 1
    for batch, state in batches:
        emit('\n' + '=' * 90)
        emit(f'{batch.name}\n运行目录：{state}')
        for stage in STAGES:
            emit(f'\n--- {stage}: {stage_status(state, stage)} ---')
            selection = load(state / f'{stage}.selection.json')
            if selection:
                emit('筛选统计：' + json.dumps({k: v for k, v in selection.items() if k != 'excluded'}, ensure_ascii=False))
                emit('排除 session 数：' + str(len(selection.get('excluded', []))))
            history = state / f'{stage}.history'
            if history.is_file():
                emit('执行历史：\n' + history.read_text(encoding='utf-8', errors='replace').rstrip())
            log = latest(list(state.glob(f'{stage}.*.log')))
            rows, tail = records(log)
            if log:
                emit(f'最新日志：{log}\n最后更新：{datetime.fromtimestamp(log.stat().st_mtime).astimezone().isoformat()}')
            # Summaries emitted by the original entry, excluding the stage wrapper.
            for r in rows:
                if 'stage' not in r and 'session' not in r and 'episode' not in r and 'video' not in r and any(k in r for k in ('total', 'total_episodes', 'done')):
                    emit('入口汇总：' + json.dumps(r, ensure_ascii=False))
            items = [r for r in rows if any(k in r for k in ('session', 'episode', 'video')) or r.get('kind') in ('result', 'worker_error')]
            counts = Counter()
            for r in items:
                status = r.get('status')
                if 'passed' in r:
                    status = 'passed' if r['passed'] else 'failed'
                counts[status or r.get('kind', 'unknown')] += 1
            if counts:
                emit('逐项结果计数（可能按 episode/video，非 session）：' + json.dumps(counts, ensure_ascii=False))
            emit('逐项结果：')
            for r in items:
                emit(json.dumps(r, ensure_ascii=False))
            # Conversion's human-readable per-session results are not JSON.
            if stage == 'convert' and log:
                with log.open(encoding='utf-8', errors='replace') as f:
                    for line in f:
                        if line.startswith(('[done]', '[failed]', '[skip-existing]')):
                            emit(line.rstrip())
            reasons = Counter()
            for session in (selection or {}).get('excluded', []):
                name = Path(session).name
                emit(f'\n排除：{session}')
                if stage == 'qc':
                    report = load(Path(session) / 'annotation/quality_report.json')
                    if report:
                        for check in report.get('checks', []):
                            if not check.get('passed') and check.get('level') == 'fail':
                                reasons[check['name']] += 1
                                emit('  ' + json.dumps(check, ensure_ascii=False))
                    else:
                        emit('  QC 报告缺失/不可读，以上阶段日志为准')
                if stage == 'convert':
                    child = output_root / batch.name / '_batch_logs' / f'{name}.log'
                    emit(f'  转换子日志（当前文件）：{child}')
                    _, child_tail = records(child if child.is_file() else None)
                    for line in child_tail:
                        emit('  ' + line)
            if reasons:
                emit('QC 失败原因计数（同 session 可多项）：' + json.dumps(reasons, ensure_ascii=False))
            if stage_status(state, stage) in ('FAILED', 'STARTED') or not rows:
                emit('日志末尾：')
                for line in tail:
                    emit(line)
    emit('\n全局 batch 状态计数：' + json.dumps(totals, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-root', type=Path, default=Path('/share/project/liuyou/humancoin/data/L1_pipeline_state'))
    p.add_argument('--output-root', type=Path, default=Path('/share/project/liuyou/humancoin/data/L1_lerobot'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not a.state_root.is_dir():
        p.error(f'State directory does not exist: {a.state_root}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', encoding='utf-8') as f:
        audit(a.state_root, a.output_root, f)
    print(f'报告已保存：{a.out}')


if __name__ == '__main__':
    main()
