"""Resume-safe term writeback, detailed task naming, and broad task classification.

Without --apply: inspect only (no API calls, annotation writes, or report creation).
Use the same report directory to resume. Inputs are frozen on the first apply.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from types import SimpleNamespace

import clean_names_and_actions_qwen as terms
import regenerate_task_names_qwen as tasks
import classify_common_tasks_qwen as common
from apply_completed_terms import snapshot, NoAPI

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def stage(folder, name, rows, function, workers, retry=False):
    """Persist each completed item before moving on; never regenerate applied tasks."""
    target = folder / (name + '.jsonl')
    saved = {}
    if target.exists():
        raw = target.read_bytes()
        offset = 0
        for line in raw.splitlines(keepends=True):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if offset + len(line) != len(raw):
                    raise
                # Remove a torn final append before writing any new records.
                with target.open('r+b') as stream:
                    stream.truncate(offset)
                break
            saved[item['path']] = item
            offset += len(line)
    pending = [r for r in rows if r['path'] not in saved or
               retry and saved[r['path']]['status'] == 'error']
    with target.open('a', encoding='utf-8') as stream, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(function, row): row for row in pending}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(path=row['path'], status='error', error=str(exc))
            saved[row['path']] = result
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps(dict(stage=name, completed=len(saved), total=len(rows),
                                  path=row['path'], status=result['status'])), flush=True)
    return [saved[r['path']] for r in rows]


def select(args):
    paths = []
    for number in range(args.start, args.end + 1):
        batch = args.root / f'batch_{number:07d}'
        if not batch.is_dir():
            raise ValueError('Missing batch: ' + str(batch))
        # Pipeline bookkeeping directories are not episode samples.
        samples = sorted(p for p in batch.iterdir()
                         if p.is_dir() and not p.name.startswith(('.', '_'))
                         and p.name != 'logs')
        if not samples:
            raise ValueError('Empty batch: ' + str(batch))
        for sample in samples:
            path = sample / 'annotation/episode_000000_structured.json'
            if not path.is_file() or path.is_symlink() or args.root not in path.resolve().parents:
                raise ValueError('Missing or unsafe annotation: ' + str(path))
            paths.append(path.relative_to(args.root).as_posix())
    return paths


def common_input(args, relative):
    data = read(args.root / relative)
    subtasks = data.get('subtasks', data.get('subtask', []))
    if not isinstance(subtasks, list):
        subtasks = [subtasks]
    return dict(dataset=args.dataset, path=relative, task_name=data.get('task_name'),
                task_description=data.get('task_description'), original_description=data.get('original_description'),
                object_names=[v.get('name') for v in data.get('objects', [])],
                subtask_texts=[json.dumps(v, ensure_ascii=False) for v in subtasks[:8] + subtasks[-2:]])


def recover_cached_terms(args, missing):
    """Reconstruct overwritten progress from existing responses; never dispatch API."""
    objects = {r['name'] for r in read(BASE/'object_vocabulary.json')['entries']}
    actions = {k:[r['verb'] for r in v] for k,v in read(BASE/'action_vocabulary.json')['groups'].items()}
    cfg = SimpleNamespace(root=args.root,state_dir=args.terms_state_dir,model=args.model,
                          max_tokens=4096,timeout=180,retry_failed=False,apply=False,dry_run=False)
    records = []
    unavailable = []
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(terms.prepare_one,args.root/p,cfg,NoAPI,objects,actions):p for p in missing}
        for future in as_completed(jobs):
            p = jobs[future]
            try:
                records.append(future.result())
            except Exception as exc:
                row = dict(path=p,status='error',error=str(exc))
                if 'Cache-only recovery: API dispatch is forbidden' in str(exc):
                    unavailable.append(row)
                else:
                    # Cached malformed/failed results remain pending, never fabricated successes.
                    records.append(row)
            if (len(records)+len(unavailable)) % 100 == 0:
                print(json.dumps(dict(stage='terms_cache_recovery',completed=len(records)+len(unavailable),
                                      total=len(missing),unavailable=len(unavailable))),flush=True)
    return records, unavailable


def run(args):
    paths = select(args)
    manifest = dict(version=1, dataset=args.dataset, root=str(args.root), start=args.start, end=args.end,
                    model=args.model, terms_state_dir=str(args.terms_state_dir), paths=paths,
                    task_max_tokens=args.task_max_tokens, batch_size=args.batch_size,
                    policy=tasks.digest(dict(task_prompt=tasks.PROMPT, common_prompt=common.PROMPT,
                                            verbs=read(BASE/'action_vocabulary.json'),
                                            taxonomy=read(BASE/'common_task_taxonomy.json'))))
    frozen = args.report_dir / 'terms_snapshot.json'
    rows = read(frozen) if frozen.exists() else snapshot(
        args.terms_state_dir / 'prepare_progress.jsonl', args.start, args.end)
    by_path = {r['path']: r for r in rows}
    missing = [p for p in paths if p not in by_path]
    recovery_path = args.report_dir/'terms_cache_recovery.json'
    if missing and recovery_path.exists():
        by_path.update({r['path']:r for r in read(recovery_path)['records'] if r['path'] in missing})
        missing = [p for p in paths if p not in by_path]
    if missing and getattr(args,'recover_cached_terms',False):
        recovered, unavailable = recover_cached_terms(args,missing)
        by_path.update({r['path']:r for r in recovered})
        if args.apply:
            terms.save_json(recovery_path,dict(records=[by_path[p] for p in paths if p in by_path],
                                               unavailable=unavailable))
        missing = [p for p in paths if p not in by_path]
    if missing:
        raise ValueError(f'{len(missing)} annotations have no completed cleaning record; '
                         'use --recover-cached-terms for overwritten progress, or wait for preparation. '
                         f'First: {missing[0]}')
    rows = [by_path[p] for p in paths]
    preview = dict(selected=len(paths), prepared=sum(r['status']=='done' for r in rows),
                   preparation_errors=sum(r['status']!='done' for r in rows), apply=args.apply)
    print(json.dumps(preview), flush=True)
    if not args.apply:
        return 0
    endpoint, overwrite = tasks.review_modules(args.argus_root)
    # Lock outside publication and share this lock across report directories.
    with overwrite.job_lock(args.root.parent / (args.root.name + '_annotation_updates')), overwrite.job_lock(args.report_dir):
        args.report_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.report_dir / 'manifest.json'
        if manifest_path.exists() and read(manifest_path) != manifest:
            raise ValueError('Report scope/policy changed; resume with the original arguments')
        terms.save_json(manifest_path, manifest)
        if not frozen.exists():
            terms.save_json(frozen, rows)
        valid = [r for r in rows if r['status']=='done']
        applied = stage(args.report_dir, 'terms_apply', valid,
                        lambda r: terms.apply_record(args.root, r, args.report_dir/'terms'), args.workers, True)
        ready = {r['path'] for r in applied if r['status']=='done'}
        verbs = {r['verb'] for r in read(BASE/'action_vocabulary.json')['groups']['Subtask_100']}
        cfg = SimpleNamespace(root=args.root, state_dir=args.report_dir/'task_state', model=args.model,
                              max_tokens=args.task_max_tokens, timeout=240, retry_failed=args.retry_failed,
                              dry_run=False, overwrite=overwrite)
        prepared = stage(args.report_dir, 'task_prepare', [r for r in valid if r['path'] in ready],
                         lambda r: tasks.prepare_one(args.root/r['path'], cfg, endpoint, verbs),
                         args.workers, args.retry_failed)
        named = stage(args.report_dir, 'task_apply', [r for r in prepared if r['status']=='done'],
                      lambda r: tasks.apply_one(args.root, r, args.report_dir/'tasks', verbs), args.workers, True)
        # Complete technical retries BEFORE freezing common evidence. Semantic review
        # keeps the original detailed task and does not block broad classification.
        errors = [r for r in applied + prepared + named if r['status']=='error']
        if errors:
            terms.save_json(args.report_dir/'pending.json', errors)
            print('Technical errors: inspect pending.json; resume with --retry-failed.', flush=True)
            return 1
        source = args.report_dir/'common_inputs.json'
        if not source.exists():
            terms.save_json(source, [common_input(args,p) for p in paths])
        inputs = read(source)
        taxonomy = read(BASE/'common_task_taxonomy.json')
        ccfg = SimpleNamespace(report_dir=args.report_dir/'common', model=args.model, max_tokens=8192,
                               timeout=240, retry_failed=args.retry_failed,
                               jianzhi_root=args.root, guanglun_root=args.root)
        # Stable packet identities; raw responses cached by the common module.
        packets = [dict(path=str(i), rows=inputs[i:i+args.batch_size])
                   for i in range(0,len(inputs),args.batch_size)]
        def classify(packet):
            result = common.classify_batch(packet['rows'], ccfg, endpoint, taxonomy)
            return dict(path=packet['path'], status='done', records=result)
        classified = stage(args.report_dir,'common_prepare',packets,classify,args.workers,args.retry_failed)
        broad = [r for packet in classified if packet['status']=='done' for r in packet['records']]
        written = stage(args.report_dir,'common_apply',broad,
                        lambda r: common.apply_one(r,ccfg,taxonomy['classes']),args.workers,True)
        pending = [r for r in rows if r['status']!='done'] + [r for r in classified+written if r['status']=='error']
        terms.save_json(args.report_dir/'pending.json',pending)
        terms.save_json(args.report_dir/'task_review.json',[r for r in prepared if r['status']=='review'])
        terms.save_json(args.report_dir/'common_other.json',[r for r in broad if r['common_task_name']=='other'])
        summary = dict(selected=len(paths),terms_applied=len(ready),task_named=len(named),
                       task_review=sum(r['status']=='review' for r in prepared),
                       common_applied=sum(r['status']=='done' for r in written),pending=len(pending))
        terms.save_json(args.report_dir/'summary.json',summary)
        print(json.dumps(summary),flush=True)
        return int(bool(pending))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['jianzhi','guanglun'],required=True)
    for key in ('root','terms-state-dir','argus-root','report-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--start',type=int,default=36)
    p.add_argument('--end',type=int,required=True)
    p.add_argument('--model',default='qwen3.7-plus')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--task-max-tokens',type=int,default=4096)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--recover-cached-terms',action='store_true',
                   help='Reconstruct missing progress from cached object/action responses without API calls')
    p.add_argument('--apply',action='store_true')
    args=p.parse_args(argv)
    for key in ('root','terms_state_dir','argus_root','report_dir'):
        setattr(args,key,getattr(args,key).resolve())
    if not 1<=args.start<=args.end or min(args.workers,args.batch_size,args.task_max_tokens)<1:
        p.error('Invalid range or worker/token/batch setting')
    if args.report_dir==args.root or args.root in args.report_dir.parents:
        p.error('Reports/backups must be outside the dataset')
    if args.report_dir==args.terms_state_dir or args.terms_state_dir in args.report_dir.parents:
        p.error('Use an independent report directory outside preparation state')
    if os.environ.get('ARGUS_ESTIMATED_CNY_LEDGER') and args.apply:
        p.error('This entry point does not implement ARGUS_ESTIMATED_CNY_LEDGER accounting')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
