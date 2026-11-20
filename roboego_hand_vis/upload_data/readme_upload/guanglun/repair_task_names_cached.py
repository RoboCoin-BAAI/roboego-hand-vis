"""Revalidate pending task names using existing response caches only; no API."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from types import SimpleNamespace

import regenerate_task_names_qwen as task

BASE = Path('/share/project/liuyou/humancoin/data')


class NoAPI:
    @staticmethod
    def settings(*args):
        raise RuntimeError('API forbidden in cached repair')


def main():
    path = BASE/'task_names_summary_01_35_20261118.json'
    summary = json.loads(path.read_text())
    task.save_json(path.with_name(path.stem+'_before_cache_repair.json'), summary)
    _, overwrite = task.review_modules(Path('/share/project/liuyou/humancoin/code/argus'))
    verbs = {r['verb'] for r in json.loads((task.BASE/'action_vocabulary.json').read_text())['groups']['Subtask_100']}
    rows = {(r['dataset'],r['path']):r for r in summary['records']}
    for name, output in [('jianzhi',BASE/'jianzhi/task_names_01_35_20261118.json'),
                         ('guanglun',BASE/'guanglun_task_names_01_35_20261118.json')]:
        root = Path(json.loads(output.read_text())['root'])
        state = output.with_name(output.stem+'_state')
        report = Path(summary['datasets'][name]['primary_report']).with_name(
                 Path(summary['datasets'][name]['primary_report']).name+'_cache_finalize')
        report.mkdir(exist_ok=False)
        selected = [r for (ds,_),r in rows.items() if ds==name and (r['status']=='error' or
                    (r['status']=='review' and r.get('review_reason') and
                     isinstance(r.get('decision'),dict) and r['decision'].get('verb') is not None))]
        def repair(old):
            source_path = root/old['path']
            data = json.loads(source_path.read_text())
            source = task.evidence(data)
            for budget in (4096,1500):
                identity = dict(source=source,before=data.get('task_name'),prompt=task.PROMPT,
                       reference=sorted(verbs),model='qwen3.7-plus',max_tokens=budget,cache_version=2)
                if 'task_name' not in data:
                    identity['before_present']=False
                response = state/'jobs'/task.digest(old['path'])[:24]/task.digest(identity)/'response.json'
                if not response.exists():
                    continue
                args=SimpleNamespace(root=root,state_dir=state,model='qwen3.7-plus',max_tokens=budget,
                     timeout=180,dry_run=False,retry_failed=False,overwrite=overwrite)
                try:
                    record=task.prepare_one(source_path,args,NoAPI,verbs)
                except Exception as exc:
                    return dict(path=old['path'],status='error',error=str(exc))
                if record['status']=='done':
                    result=task.apply_one(root,record,report,verbs)
                    return dict(record,changed=result['changed'],added=not record.get('before_present',True))
                return dict(record,changed=False,added=False)
            return dict(path=old['path'],status='cache_missing',error='No matching cache for current evidence')
        repaired=[]
        with ThreadPoolExecutor(16) as pool:
            futures={pool.submit(repair,r):r for r in selected}
            for future in as_completed(futures):
                old=futures[future]
                try:
                    record=future.result()
                except Exception as exc:
                    record=dict(path=old['path'],status='error',error=str(exc))
                repaired.append(record)
                if record['status']=='cache_missing':
                    continue
                updated=dict(old,status=record['status'],changed=record.get('changed',False),
                    added=record.get('added',False),task_name=record.get('after') if record['status']=='done' else old.get('before'),
                    decision=record.get('decision'),error=record.get('error'),review_reason=record.get('review_reason'))
                rows[(name,old['path'])]=updated
        task.save_json(report/'repair_report.json',dict(model_requests=0,records=repaired))
        final=[r for (ds,_),r in rows.items() if ds==name]
        summary['datasets'][name].update(done=sum(r['status']=='done' for r in final),
            changed=sum(r['changed'] for r in final),added=sum(r.get('added',False) for r in final),
            review=sum(r['status']=='review' for r in final),errors=sum(r['status']=='error' for r in final),
            cache_repair_report=str(report))
        print(name,json.dumps(summary['datasets'][name]),flush=True)
    summary['records']=sorted(rows.values(),key=lambda r:(r['dataset'],r['path']))
    summary['cache_repair_model_requests']=0
    task.save_json(path,summary)
    task.save_json(BASE/'task_names_review_01_35_20261118.json',[r for r in summary['records'] if r['status']!='done'])


if __name__=='__main__':
    main()
