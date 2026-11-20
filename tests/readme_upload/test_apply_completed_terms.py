import importlib.util
import json
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parents[2] / 'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('apply_completed_test', DIRECTORY/'apply_completed_terms.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_apply_range_preserves_pending_and_later_batches(tmp_path):
    root = tmp_path/'data'
    state = tmp_path/'state'
    state.mkdir()
    rows, paths = [], []
    for batch in (1, 2, 36):
        path = root/f'batch_{batch:07d}/sample/annotation/episode_000000_structured.json'
        path.parent.mkdir(parents=True)
        data = dict(objects=[dict(name='red cup')], actions=dict(vocabulary=['rewash']),
                    task_description='Wash cup', task_name='wash cup',
                    scene_classification=dict(category='household', area='kitchen'))
        path.write_text(json.dumps(data))
        paths.append(path)
        row = dict(path=path.relative_to(root).as_posix(), status='done',
                   before=m.combined.source_fields(data), after=dict(object_names=['cup'], action_vocabulary=['wash']))
        if batch == 2:
            row = dict(path=row['path'], status='error', error='Missing response')
        rows.append(row)
    originals = [p.read_bytes() for p in paths]
    (state/'prepare_progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"partial":')
    report = tmp_path/'report'
    assert m.main(['--root',str(root),'--state-dir',str(state),'--report-dir',str(report)]) == 0
    latest = json.loads(paths[0].read_bytes())
    assert latest['objects'][0]['name'] == 'cup'
    assert latest['actions']['vocabulary'] == ['wash']
    assert latest['task_name'] == 'wash cup'
    assert paths[1].read_bytes() == originals[1]
    assert paths[2].read_bytes() == originals[2]
    summary = json.loads((report/'apply_report.json').read_text())['summary']
    assert summary['selected'] == 2 and summary['pending_preparation'] == 1
    assert summary['changed'] == 1 and summary['write_errors'] == 0
