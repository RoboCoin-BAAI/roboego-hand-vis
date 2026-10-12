from __future__ import annotations

import json

import pytest
from pathlib import Path

from roboego_hand_vis.post_annotation_actions import compute
from roboego_hand_vis.post_annotation_actions.compute import PostAnnotationActionsConfig, run_post_annotation_actions


@pytest.mark.parametrize("review_count", [0, 1])
def test_run_post_annotation_actions_writes_subtasks_after_merged_actions(tmp_path, monkeypatch, review_count) -> None:
    qwen_chunk_dir = tmp_path / "qwen_frame_annotation_chunks"
    qwen_chunk_dir.mkdir()
    (qwen_chunk_dir / "chunk_0000.json").write_text("{}", encoding="utf-8")

    def fake_write_stage1_action_output(*, input_json: Path, output_json: Path) -> dict:
        payload = {
            "stage": "stage1_action_extraction",
            "frames": [
                {
                    "t": 0,
                    "scene": "white table with a black rack",
                    "activity": "wearer adjusts a rack",
                    "frame_actions": [{"action": "operate", "object_name": "rack", "confidence": 0.9}],
                }
            ],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_write_text_resolver_review_input(*, input_json: Path, output_json: Path) -> dict:
        payload = {"frame_count": review_count, "frames": [{"t": 0}] if review_count else []}
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_text_resolver(**kwargs):
        assert kwargs["base_url"] == "http://model.example/v1/chat/completions"
        return {"status": "ok", "frames": []}

    def fake_standardize_stage1_actions(payload: dict, **kwargs) -> dict:
        assert kwargs["base_url"] == "http://model.example/v1/chat/completions"
        return {
            "stage": "robolabel_stage1_actions",
            "frames": [
                {
                    "t": 0,
                    "scene": "white table with a black rack",
                    "activity": "wearer adjusts a rack",
                    "frame_actions": [
                        {
                            "action": "manipulate",
                            "phase_action": "rotate",
                            "skill": "manipulate",
                            "object_name": "rack",
                            "object_description": "black metal two-tier rack",
                            "confidence": 0.9,
                        }
                    ],
                }
            ],
        }

    def fake_subtask_resolver(review_payload: dict, **kwargs) -> dict:
        assert kwargs["base_url"] == "http://model.example/v1/chat/completions"
        assert review_payload["frames"] == [
            {"t": 0, "scene": "white table with a black rack", "activity": "wearer adjusts a rack"}
        ]
        return {
            "status": "ok",
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 0,
                    "end_ts": 0,
                    "start_frame": 0,
                    "end_frame": 0,
                    "source_frames": [0],
                }
            ],
        }

    def fake_object_resolver(review_payload: dict, **kwargs) -> dict:
        assert kwargs["base_url"] == "http://model.example/v1/chat/completions"
        assert review_payload["subtasks"][0]["objects"] == [
            {"name": "rack", "descriptions": ["black metal two-tier rack"]}
        ]
        return {
            "status": "ok",
            "object_aliases_by_subtask": {
                "1": [{"canonical_name": "rack", "aliases": ["rack", "black metal two-tier rack"]}]
            },
        }

    monkeypatch.setattr(compute, "write_stage1_action_output", fake_write_stage1_action_output)
    monkeypatch.setattr(compute, "write_text_resolver_review_input", fake_write_text_resolver_review_input)
    monkeypatch.setattr(compute, "call_qwen_text_resolver", fake_text_resolver)
    monkeypatch.setattr(compute, "standardize_stage1_actions", fake_standardize_stage1_actions)
    monkeypatch.setattr(compute, "call_qwen_subtask_resolver", fake_subtask_resolver)
    monkeypatch.setattr(compute, "call_qwen_object_resolver", fake_object_resolver)

    output_dir = tmp_path / "annotation"
    config = PostAnnotationActionsConfig(
        qwen_chunk_dir=qwen_chunk_dir,
        action_chunk_dir=output_dir / "stage1_action_chunks",
        review_chunk_dir=output_dir / "stage1_action_text_review_chunks",
        resolved_chunk_dir=output_dir / "stage1_action_resolved_chunks",
        merged_output_json=output_dir / "stage1_actions_left_v1_resolved.json",
        subtask_output_json=output_dir / "stage1_subtasks_left_v1.json",
        frames_dir=None,
        text_model="qwen-plus",
        subtask_model="qwen-plus",
        subtask_prompt=Path("docs/prompts/stage1_subtask_segmentation_en.md"),
        subtask_object_prompt=Path("docs/prompts/stage1_subtask_object_resolution_en.md"),
        subtask_chunk_size=8,
        subtask_overlap=1,
        generate_subtasks=True,
        use_vl_object_resolver=False,
        vl_object_model="qwen-vl-plus",
        max_vl_frames_per_chunk=30,
        api_key_env="DASHSCOPE_API_KEY",
        timeout_s=120.0,
        overwrite=True,
        base_url="http://model.example/v1/chat/completions",
    )

    result = run_post_annotation_actions(config)

    assert result["subtask_output_json"] == str(config.subtask_output_json)
    subtask_payload = json.loads(config.subtask_output_json.read_text(encoding="utf-8"))
    assert subtask_payload["subtask_count"] == 1
    assert subtask_payload["subtasks"][0]["action"] == "manipulate"
    assert subtask_payload["subtasks"][0]["object_name"] == "rack"
