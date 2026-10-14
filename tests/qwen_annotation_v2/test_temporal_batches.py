import json
import re
import threading
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from roboego_hand_vis.qwen_annotation_v2 import cli, compute
from roboego_hand_vis.qwen_annotation_v2.frame_annotation import extract_video_frames, extract_video_frames_from_windows


def item(t):
    return dict(t=t, scene='workspace', activity='wearer holds an object', hand_visible=False,
                hands_visible={'left': False, 'right': False}, wearer_region={
                    'visible': False, 'bbox': None, 'body_parts': [], 'confidence': 0., 'notes': ''},
                hand_object_interaction=False, interaction_objects=[], interaction_object_attributes=[],
                attention_objects=[], task_context=False, task_objects=[], privacy=False,
                privacy_types=[], image_quality='clear', uncertainty=0., notes='')


def setup(tmp_path, monkeypatch, **options):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'video')
    paths = []
    for t in range(6):
        p = tmp_path / f'image_{t}.jpg'
        cv2.imwrite(str(p), np.full((16, 24, 3), t * 30, np.uint8))
        paths.append(p)
    monkeypatch.setattr(compute, 'extract_video_frames', lambda **kwargs: (paths, list(range(6))))
    config = cli.config_from_yaml({'parameters': {'batch_size': 2, 'max_retries': 0, **options}},
                                 video=video, output_json=tmp_path / 'annotation/out.json')
    return config


def test_fractional_samples_have_unique_images_and_downstream_lookup(tmp_path):
    video = tmp_path / 'video.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 4, (32, 24))
    assert writer.isOpened()
    for k in range(12):
        writer.write(np.full((24, 32, 3), k * 18, np.uint8))
    writer.release()
    paths, times = extract_video_frames(video_path=video, output_dir=tmp_path / 'frames',
                                       start_s=0, duration_s=None, fps=2, max_width=32)
    assert len(paths) == len(set(paths)) == 6
    assert times == [0., .5, 1., 1.5, 2., 2.5]
    from roboego_hand_vis.qwen_annotation_v2.sampled_frames import sampled_frame_name
    assert [tmp_path / 'frames' / sampled_frame_name(t) for t in times] == paths
    assert len({p.read_bytes() for p in paths}) == 6


def test_overlapping_source_windows_return_increasing_times(tmp_path):
    video = tmp_path / 'video.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 4, (32, 24))
    for k in range(12):
        writer.write(np.full((24, 32, 3), k * 18, np.uint8))
    writer.release()
    paths, times = extract_video_frames_from_windows(video_path=video, output_dir=tmp_path / 'frames',
        windows=[{'start': 0, 'end': 2}, {'start': .5, 'end': 2.5}], fps=1, max_width=32)
    assert times == sorted(times)
    assert len(paths) == len(set(paths))


def test_parallel_overlapping_requests_save_only_owned_frames_in_time_order(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch, overlap_frames=1, max_concurrency=2)
    barrier = threading.Barrier(2)
    seen = []
    def call(**kwargs):
        times = [float(re.search(r't=([0-9.]+)', block['text'])[1])
                 for block in kwargs['messages'][1]['content'] if block.get('text', '').startswith('Frame timestamp')]
        seen.append(times)
        if len(seen) <= 2:
            barrier.wait(timeout=3)
        text = json.dumps({'frames': [item(t) for t in reversed(times)]})
        result = kwargs['response_parser'](text)
        return result | {'raw_response': text, 'usage': {'total_tokens': 10}}
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', call)
    summary = compute.run_qwen_annotation(config)
    assert sorted(seen) == [[0., 1., 2.], [1., 2., 3., 4.], [3., 4., 5.]]
    assert summary['status'] == 'complete'
    assert summary['max_concurrency'] == 2
    assert [f['t'] for f in json.loads(config.output_json.read_text())['frames']] == list(range(6))
    assert [f['t'] for f in json.loads(next(config.batch_output_dir.glob('batch_0001*.json')).read_text())['frames']] == [2., 3.]


@pytest.mark.parametrize('times', [[0], [0, 0, 1], [0, 1, 99], [float('nan'), 1]])
def test_incomplete_duplicate_or_unrequested_timestamps_fail_a_batch(tmp_path, monkeypatch, times):
    config = setup(tmp_path, monkeypatch, overlap_frames=1)
    def call(**kwargs):
        text = json.dumps({'frames': [item(t) for t in times]})
        return kwargs['response_parser'](text) | {'raw_response': text}
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', call)
    result = compute.run_qwen_annotation(config)
    assert result['failed_batch_count'] >= 1
    assert result['status'] != 'complete'


def test_resume_accepts_concurrency_change_but_rejects_overlap_change_before_calls(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch, overlap_frames=1)
    def call(**kwargs):
        times = [float(re.search(r't=([0-9.]+)', block['text'])[1])
                 for block in kwargs['messages'][1]['content'] if block.get('text', '').startswith('Frame timestamp')]
        text = json.dumps({'frames': [item(t) for t in times]})
        return kwargs['response_parser'](text) | {'raw_response': text}
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', call)
    assert compute.run_qwen_annotation(config)['status'] == 'complete'
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', lambda **kwargs: pytest.fail('must not request completed batches'))
    assert compute.run_qwen_annotation(replace(config, max_concurrency=2))['status'] == 'complete'
    original = config.output_json.read_bytes()
    with pytest.raises(ValueError, match='fresh output'):
        compute.run_qwen_annotation(replace(config, overlap_frames=2))
    assert config.output_json.read_bytes() == original


def test_distinct_valid_windows_are_not_joined_even_when_gap_is_small():
    from roboego_hand_vis.qwen_annotation_v2.temporal_batches import plan_batches
    batches = plan_batches([0., 1.2], 10, 2, 1,
                           windows=[{'start': 0, 'end': .6}, {'start': 1.2, 'end': 2}])
    assert [batch.targets for batch in batches] == [(0,), (1,)]
    assert [batch.context for batch in batches] == [(0,), (1,)]


def test_unbound_legacy_results_are_rejected_before_requests(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch, batch_size=10)
    config.output_json.parent.mkdir()
    config.output_json.write_text(json.dumps({'source_video': str(config.video),
                                            'prompt': str(config.prompt), 'frames': [item(0)]}))
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', lambda **kwargs: pytest.fail('no request'))
    with pytest.raises(ValueError, match='fresh output'):
        compute.run_qwen_annotation(config)


def test_resume_restores_missing_batch_artifact_without_repeating_other_batches(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch, overlap_frames=1)
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        times = [float(re.search(r't=([0-9.]+)', block['text'])[1])
                 for block in kwargs['messages'][1]['content'] if block.get('text', '').startswith('Frame timestamp')]
        text = json.dumps({'frames': [item(t) for t in times]})
        return kwargs['response_parser'](text) | {'raw_response': text}
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', call)
    assert compute.run_qwen_annotation(config)['status'] == 'complete'
    next(config.batch_output_dir.glob('batch_0001*.json')).unlink()
    summary = compute.run_qwen_annotation(config)
    assert summary['status'] == 'complete'
    assert summary['request_count'] == 1
    assert len(calls) == 4


def test_raw_recovery_filters_context_without_duplicate_timestamps(tmp_path, monkeypatch):
    import shutil
    config = setup(tmp_path, monkeypatch, overlap_frames=1)
    def call(**kwargs):
        times = [float(re.search(r't=([0-9.]+)', block['text'])[1])
                 for block in kwargs['messages'][1]['content'] if block.get('text', '').startswith('Frame timestamp')]
        text = json.dumps({'frames': [item(t) for t in times]})
        return kwargs['response_parser'](text) | {'raw_response': text}
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', call)
    assert compute.run_qwen_annotation(config)['status'] == 'complete'
    config.checkpoint_json.unlink()
    shutil.rmtree(config.batch_output_dir)
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', lambda **kwargs: pytest.fail('raw recovery should suffice'))
    summary = compute.run_qwen_annotation(replace(config, recover_from_raw=True))
    assert summary['status'] == 'complete'
    assert summary['request_count'] == 0
    assert [f['t'] for f in json.loads(config.output_json.read_text())['frames']] == list(range(6))


def test_timestamp_validation_uses_existing_http_retry_feedback(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch, batch_size=6, max_retries=1,
                   api_key_env='TEST_V2_KEY', retry_sleep_s=0, max_concurrency=1)
    requests = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            times = [0] if len(requests) == 1 else list(range(6))
            return json.dumps({'choices': [{'message': {'content': json.dumps({'frames': [item(t) for t in times]})}}]}).encode()
    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return Response()
    monkeypatch.setenv('TEST_V2_KEY', 'test-only')
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    result = compute.run_qwen_annotation(config)
    assert result['status'] == 'complete'
    assert len(requests) == 2
    assert 'Missing target frame timestamps' in requests[1]['messages'][1]['content'][-1]['text']
    assert len(json.loads(config.output_json.read_text())['frames']) == 6


def test_one_failed_batch_does_not_lose_others_and_resume_requests_only_failed(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch, overlap_frames=1, max_concurrency=2)
    failing = True
    calls = []
    def call(**kwargs):
        targets = json.loads(re.search(r'Target timestamps to annotate exactly once: (\[.*?\])',
                                      kwargs['messages'][1]['content'][0]['text'])[1])
        calls.append(targets)
        if failing and targets == [2, 3]:
            return {'status': 'timeout_error', 'raw_response': 'test timeout'}
        text = json.dumps({'frames': [item(t) for t in targets]})
        return kwargs['response_parser'](text) | {'raw_response': text}
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', call)
    assert compute.run_qwen_annotation(config)['status'] == 'partial_failed'
    assert [f['t'] for f in json.loads(config.output_json.read_text())['frames']] == [0, 1, 4, 5]
    failing = False
    assert compute.run_qwen_annotation(config)['status'] == 'complete'
    assert calls.count([2, 3]) == 2
    assert calls.count([0, 1]) == calls.count([4, 5]) == 1


def test_complete_checkpoint_rebuilds_missing_chunks_without_model_calls(tmp_path, monkeypatch):
    import shutil
    config = setup(tmp_path, monkeypatch, overlap_frames=1)
    def call(**kwargs):
        targets = json.loads(re.search(r'Target timestamps to annotate exactly once: (\[.*?\])',
                                      kwargs['messages'][1]['content'][0]['text'])[1])
        text = json.dumps({'frames': [item(t) for t in targets]})
        return kwargs['response_parser'](text) | {'raw_response': text}
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', call)
    assert compute.run_qwen_annotation(config)['status'] == 'complete'
    shutil.rmtree(config.chunk_output_dir)
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', lambda **kw: pytest.fail('must reuse complete checkpoint'))
    assert compute.run_qwen_annotation(config)['status'] == 'complete'
    chunks = list(config.chunk_output_dir.glob('chunk_*.json'))
    assert len(chunks) == 1
    assert json.loads(chunks[0].read_text())['frame_count'] == 6
