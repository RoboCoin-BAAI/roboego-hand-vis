from __future__ import annotations

import json
from pathlib import Path

from roboego_hand_vis.qwen_annotation import cli
from roboego_hand_vis.qwen_annotation.frame_annotation import call_qwen_frame_annotations


def test_config_from_yaml_reads_local_vllm_parameters(tmp_path: Path) -> None:
    video = tmp_path / "left.mkv"
    output_json = tmp_path / "annotation" / "qwen.json"

    config = cli.config_from_yaml(
        {
            "parameters": {
                "model": "qwen3.8-27b",
                "api_key_env": "QWEN_API_KEY",
                "base_url": "http://127.0.0.1:18000/v1/chat/completions",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        },
        video=video,
        output_json=output_json,
    )

    assert config.model == "qwen3.8-27b"
    assert config.api_key_env == "QWEN_API_KEY"
    assert config.base_url == "http://127.0.0.1:18000/v1/chat/completions"
    assert config.chat_template_kwargs == {"enable_thinking": True}


def test_call_qwen_frame_annotations_sends_local_vllm_body(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            content = {
                "frames": [
                    {
                        "t": 0,
                        "scene": "test scene",
                        "activity": "wearer observes a shelf",
                        "hand_visible": False,
                        "hands_visible": {"left": False, "right": False},
                        "wearer_region": {
                            "visible": False,
                            "bbox": None,
                            "body_parts": [],
                            "confidence": 0.0,
                            "notes": "",
                        },
                        "hand_object_interaction": False,
                        "interaction_objects": [],
                        "interaction_object_attributes": [],
                        "attention_objects": [],
                        "task_context": True,
                        "task_objects": [],
                        "privacy": False,
                        "privacy_types": [],
                        "image_quality": "clear",
                        "uncertainty": 0.0,
                        "notes": "",
                    }
                ]
            }
            payload = {
                "id": "chatcmpl-test",
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"total_tokens": 10},
            }
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["full_url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("QWEN_API_KEY", "sk-qwen-local")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    parsed = call_qwen_frame_annotations(
        messages=[{"role": "user", "content": "hello"}],
        model="qwen3.8-27b",
        api_key_env="QWEN_API_KEY",
        base_url="http://127.0.0.1:18000/v1/chat/completions",
        chat_template_kwargs={"enable_thinking": True},
    )

    assert parsed["status"] == "ok"
    assert captured["full_url"] == "http://127.0.0.1:18000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-qwen-local"
    assert captured["body"]["model"] == "qwen3.8-27b"
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": True}
