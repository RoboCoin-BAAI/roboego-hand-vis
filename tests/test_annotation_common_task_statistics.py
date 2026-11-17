import csv
import json

from roboego_hand_vis.annotation_statistics.run import collect, extract


def test_common_task_counts_missing_and_details(tmp_path):
    root = tmp_path/'dataset'
    docs = [{'common_task': 'organize document'}, {'common_task': 'organize document'},
            {'common_task': 'repair cable'}, {'common_task': None}, {}, {'common_task': ''}]
    for index, document in enumerate(docs):
        directory = root/'batch_0000001'/str(index)/'annotation'
        directory.mkdir(parents=True)
        (directory/'episode_000000_structured.json').write_text(json.dumps(document))
    output = tmp_path/'report'
    summary = collect(root, output, 'episode_000000_structured.json')
    assert summary['fields']['common_task'] == {
        'unique_values': 3, 'occurrences': 4, 'missing_or_blank_files': 3}
    with (output/'common_task_counts.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0] == {'value': 'organize document', 'occurrences': '2', 'episode_count': '2'}
    assert (output/'common_task_mapping.csv').exists()
    assert json.loads((output/'episodes.jsonl').read_text().splitlines()[0])['common_task'] == ['organize document']
    assert extract({'common_task': 123})[1] == ['common_task: expected string']
