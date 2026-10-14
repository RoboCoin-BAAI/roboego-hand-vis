import base64
import json

import cv2
import numpy as np
import pytest


def test_grid_groups_chronological_images_and_embeds_timestamps(tmp_path):
    from roboego_hand_vis.qwen_annotation_v2.grid_messages import build_messages
    paths = []
    for i in range(5):
        path = tmp_path / f'{i}.jpg'
        cv2.imwrite(str(path), np.full((24, 40, 3), i * 40, np.uint8))
        paths.append(path)
    messages = build_messages(system_prompt='system', user_prompt='target frames', frame_paths=paths,
                              timestamps_s=[0, .5, 1, 1.5, 2], image_mode='grid',
                              grid_columns=4, grid_cell_width=64)
    content = messages[1]['content']
    pictures = [part for part in content if part['type'] == 'image_url']
    assert len(pictures) == 2
    assert any('0.500000' in part.get('text', '') for part in content)
    data = base64.b64decode(pictures[0]['image_url']['url'].split(',')[1])
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    # Cells are chronological and colors remain in the matching column.
    assert image[-8:, :40].mean() < image[-8:, 132:172].mean()


def test_individual_mode_preserves_one_picture_per_frame(tmp_path):
    from roboego_hand_vis.qwen_annotation_v2.grid_messages import build_messages
    path = tmp_path / 'frame.jpg'
    cv2.imwrite(str(path), np.zeros((24, 40, 3), np.uint8))
    messages = build_messages(system_prompt='system', user_prompt='target', frame_paths=[path, path],
                              timestamps_s=[0, 1], image_mode='individual',
                              grid_columns=4, grid_cell_width=512)
    assert len([p for p in messages[1]['content'] if p['type'] == 'image_url']) == 2


def test_switching_image_mode_requires_new_output_directory(tmp_path, monkeypatch):
    from roboego_hand_vis.qwen_annotation_v2.cli import config_from_yaml
    from roboego_hand_vis.qwen_annotation_v2.run_settings import bind_run_settings
    from dataclasses import replace
    video = tmp_path / 'video.mkv'; video.touch()
    config = config_from_yaml({}, video=video, output_json=tmp_path / 'annotation/out.json')
    bind_run_settings(config)
    with pytest.raises(ValueError, match='fresh output'):
        bind_run_settings(replace(config, image_mode='grid'))


def test_grid_pipeline_uses_target_timestamps_and_preserves_frame_schema(tmp_path, monkeypatch):
    from roboego_hand_vis.qwen_annotation_v2 import cli, compute
    video = tmp_path / 'video.mkv'; video.touch()
    image = tmp_path / 'sample.jpg'
    cv2.imwrite(str(image), np.zeros((24, 40, 3), np.uint8))
    monkeypatch.setattr(compute, 'extract_video_frames', lambda **kw: ([image] * 3, [0., 1., 2.]))
    config = cli.config_from_yaml({'parameters': {'image_mode': 'grid', 'max_concurrency': 1}},
                                 video=video, output_json=tmp_path / 'annotation/out.json')
    seen = []
    def call(**kwargs):
        from roboego_hand_vis.qwen_annotation.frame_annotation import build_data_inspection_placeholder_frames
        frames = build_data_inspection_placeholder_frames([0., 1., 2.], {})
        text = json.dumps({'frames': frames})
        seen.extend(kwargs['messages'][1]['content'])
        return kwargs['response_parser'](text) | {'raw_response': text}
    monkeypatch.setattr(compute, 'call_qwen_frame_annotations', call)
    result = compute.run_qwen_annotation(config)
    assert result['status'] == 'complete'
    assert result['image_mode'] == 'grid'
    assert len([block for block in seen if block['type'] == 'image_url']) == 1
    assert [frame['t'] for frame in json.loads(config.output_json.read_text())['frames']] == [0., 1., 2.]


def test_partial_failure_cli_exits_nonzero(tmp_path, monkeypatch):
    from roboego_hand_vis.qwen_annotation_v2 import cli
    import sys
    monkeypatch.setattr(sys, 'argv', ['run', '--video', str(tmp_path / 'video.mkv'),
                                     '--output_json', str(tmp_path / 'new/out.json')])
    monkeypatch.setattr(cli, 'run_qwen_annotation', lambda config: {'status': 'partial_failed'})
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
