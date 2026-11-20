"""Finish the authorized 2026-11-18 task-name jobs; retains all primary reports."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from types import SimpleNamespace

import regenerate_task_names_qwen as task

BASE = Path('/share/project/liuyou/humancoin/data')
JOBS = [
    ('jianzhi', BASE/'jianzhi/task_names_01_35_20261118.json', BASE/'jianzhi/task_names_apply_01_35_20261118'),
    ('guanglun', BASE/'guanglun_task_names_01_35_20261118.json', BASE/'guanglun_task_names_apply_01_35_20261118'),
]


def main():
    endpoint, overwrite = task.review_modules(Path('/share/project/liuyou/humancoin/code/argus'))
    endpoint.settings('qwen3.7-plus')
    verbs = {r['verb'] for r in json.loads((task.BASE/'action_vocabulary.json').read_text())['groups']['Subtask_100']}
    summaries, all_results = {}, []
    for dataset, bundle_path, primary_report_dir in JOBS:
        while not (primary_report_dir/'apply_report.json').exists():
            time.sleep(5)
        bundle = json.loads(bundle_path.read_text())
        if bundle['status'] != 'complete':
            raise ValueError('Primary generation did not finish')
        root = Path(bundle['root'])
        state = bundle_path.with_name(bundle_path.stem+'_state')
        recovery_dir = primary_report_dir.with_name(primary_report_dir.name+'_recovery')
        recovery_dir.mkdir(parents=True, exist_ok=False)
        original = {r['path']: r for r in bundle['records']}
        applied = {r['path']: r for r in json.loads((primary_report_dir/'apply_report.json').read_text())['records']}
        failed = [r for r in bundle['records'] if r['status']=='error']
        task.save_json(recovery_dir/'selected_errors.json', failed)
        recovered = []
        def generate(row):
            # Complete cached semantic responses are revalidated without rebilling.
            # Only truncated/broken responses use a fresh, larger output budget.
            larger = 'Incomplete' in row.get('error','') or 'Extra data' in row.get('error','') or 'Expecting' in row.get('error','')
            args = SimpleNamespace(root=root, state_dir=state, model='qwen3.7-plus',
                    max_tokens=4096 if larger else 1500, timeout=180,
                    dry_run=False, retry_failed=False, overwrite=overwrite)
            return task.prepare_one(root/row['path'], args, endpoint, verbs)
        with (recovery_dir/'progress.jsonl').open('w') as progress, ThreadPoolExecutor(16) as pool:
            futures = {pool.submit(generate, row): row for row in failed}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    row = dict(path=futures[future]['path'], status='error', error=str(exc))
                recovered.append(row)
                progress.write(json.dumps(row,ensure_ascii=False)+'\n'); progress.flush()
                print(json.dumps(dict(dataset=dataset,phase='recover',completed=len(recovered),total=len(failed),status=row['status'])),flush=True)
        task.save_json(recovery_dir/'generated.json', dict(records=recovered))
        recovery_applied = []
        for row in recovered:
            if row['status']=='done':
                try:
                    result = task.apply_one(root, row, recovery_dir, verbs)
                except Exception as exc:
                    result = dict(path=row['path'],status='error',error=str(exc))
            else:
                result = dict(path=row['path'],status=row['status'],changed=False,error=row.get('error'))
            recovery_applied.append(result)
            original[row['path']] = row
            applied[row['path']] = result
        task.save_json(recovery_dir/'apply_report.json',dict(records=recovery_applied))
        final = []
        for path, row in sorted(original.items()):
            result = applied[path]
            status = result['status']
            final.append(dict(dataset=dataset,path=path,status=status,
                 before=row.get('before'),task_name=row.get('after') if status=='done' else row.get('before'),
                 changed=result.get('changed',False),added=status=='done' and row.get('before_present',True) is False,
                 task_description=row.get('source',{}).get('task_description'),
                 decision=row.get('decision'),review_reason=row.get('review_reason'),
                 error=result.get('error') or row.get('error')))
        summary = dict(selected=len(final),done=sum(r['status']=='done' for r in final),
                       changed=sum(r['changed'] for r in final),added=sum(r['added'] for r in final),
                       review=sum(r['status']=='review' for r in final),errors=sum(r['status']=='error' for r in final),
                       primary_report=str(primary_report_dir),recovery_report=str(recovery_dir))
        task.save_json(recovery_dir/'final_report.json',dict(summary=summary,records=final))
        summaries[dataset] = summary
        all_results.extend(final)
        print(json.dumps(dict(dataset=dataset,summary=summary)),flush=True)
    summary_path = BASE/'task_names_summary_01_35_20261118.json'
    task.save_json(summary_path,dict(status='complete',scope=['task_name'],start=1,end=35,
                   model='qwen3.7-plus',excluded_terms_failures=58,datasets=summaries,
                   records=all_results))
    task.save_json(BASE/'task_names_review_01_35_20261118.json',
                   [r for r in all_results if r['status']!='done'])
    print('COMPLETE',summary_path,flush=True)


if __name__=='__main__':
    main()
