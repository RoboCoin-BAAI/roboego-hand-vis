"""Merge dataset-local common labels with fewer than five samples."""
import json
import stat
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from apply_scene_classification import atomic_write
from classify_common_tasks_qwen import patch_common
from reclassify_scenes_qwen import save_json

BASE = Path('/share/project/liuyou/humancoin/data')
ROOTS = {'jianzhi': BASE/'jianzhi/wait_for_publish', 'guanglun': BASE/'guanglun_depth'}
REPORT = BASE/'common_verb_lt5_apply_20261119'

def read(row):
    root = ROOTS[row['dataset']].resolve()
    path = root/row['path']
    if path.is_symlink() or root not in path.resolve().parents:
        raise ValueError('Unsafe path: '+str(path))
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    name = data['common_task_name']
    if not isinstance(name, str) or not name.endswith('_task'):
        raise ValueError('Invalid label: '+str(path))
    return row, path, raw, data

def main():
    previous = json.loads((BASE/'common_verb_round2_apply_20261119/apply_report.json').read_text())
    rows = previous['records']
    assert len(rows) == 16259 and all(r['status']=='done' for r in rows)
    assert len({(r['dataset'],r['path']) for r in rows}) == len(rows)
    with ThreadPoolExecutor(16) as pool:
        snapshots = list(pool.map(read, rows))
    counts = {ds:Counter(data['common_task_name'] for row,_,_,data in snapshots if row['dataset']==ds) for ds in ROOTS}
    rare = {ds:{name:n for name,n in c.items() if n<5 and name!='other_task'} for ds,c in counts.items()}
    REPORT.mkdir(parents=True,exist_ok=True)
    save_json(REPORT/'rare_labels.json',rare)
    def update(item):
        row,path,raw,data = item
        old = data['common_task_name']
        name = 'other_task' if old in rare[row['dataset']] else old
        try:
            revised = patch_common(raw,name) if old!=name else raw
            if path.read_bytes()!=raw:
                raise ValueError('Concurrent modification')
            if old!=name:
                atomic_write(path,revised,stat.S_IMODE(path.stat().st_mode))
            actual = path.read_bytes()
            if actual!=revised or json.loads(actual.decode('utf-8-sig'))!=dict(data,common_task_name=name):
                raise ValueError('Verification failed')
            return dict(dataset=row['dataset'],path=row['path'],status='done',changed=old!=name,previous_common_task_name=old,common_task_name=name,byte_delta=len(actual)-len(raw))
        except Exception as exc:
            return dict(dataset=row['dataset'],path=row['path'],status='error',error=str(exc))
    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(update,snapshots))
    summary = dict(annotations=len(results),changed=sum(r.get('changed',False) for r in results),errors=sum(r['status']=='error' for r in results),threshold='count < 5 per dataset',common_types=len({r['common_task_name'] for r in results if r['status']=='done'}),datasets={ds:dict(changed=sum(r.get('changed',False) for r in results if r['dataset']==ds),rare_types=len(rare[ds]),common_types=len({r['common_task_name'] for r in results if r['dataset']==ds and r['status']=='done'})) for ds in ROOTS})
    save_json(REPORT/'apply_report.json',dict(summary=summary,records=results))
    save_json(REPORT/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return int(summary['errors']>0)

if __name__=='__main__':
    raise SystemExit(main())
