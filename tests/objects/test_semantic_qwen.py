import base64
import json
from pathlib import Path

import pytest

from roboego_hand_vis.objects.semantic_qwen import (
    build_qwen_messages,
    call_qwen_semantics,
    parse_semantic_response,
    select_key_frames,
)
from roboego_hand_vis.objects.segmentation import ObjectObservation


def _valid_semantic_response(**overrides: object) -> str:
    payload = {
        "category": "box",
        "color": "yellow",
        "shape": "rectangular",
        "material": "paper",
        "visual_description": "yellow box",
        "confidence": 0.8,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_semantic_response_accepts_json_object():
    parsed = parse_semantic_response(
        '{"category":"box","color":"yellow","shape":"rectangular","material":"paper","visual_description":"yellow box","confidence":0.8}'
    )
    assert parsed["category"] == "box"
    assert parsed["confidence"] == 0.8


def test_parse_semantic_response_extracts_json_from_text():
    parsed = parse_semantic_response(
        'result: {"category":"toy","color":"yellow","shape":"box","material":"plastic","visual_description":"toy","confidence":0.6}'
    )
    assert parsed["material"] == "plastic"


def test_select_key_frames_uses_first_largest_and_spaced_frames(tmp_path: Path):
    observations = [
        ObjectObservation(0, "yellow_object_0", [0, 0, 10, 10], 100, "detected", crop_path=tmp_path / "0.png"),
        ObjectObservation(10, "yellow_object_0", [0, 0, 30, 30], 900, "detected", crop_path=tmp_path / "10.png"),
        ObjectObservation(20, "yellow_object_0", [0, 0, 20, 20], 400, "detected", crop_path=tmp_path / "20.png"),
    ]
    assert select_key_frames(observations, max_frames=3) == [0, 10, 20]


def test_select_key_frames_with_zero_limit_returns_no_frames(tmp_path: Path):
    observations = [
        ObjectObservation(0, "yellow_object_0", [0, 0, 10, 10], 100, "detected", crop_path=tmp_path / "0.png"),
    ]
    assert select_key_frames(observations, max_frames=0) == []


def test_select_key_frames_with_one_limit_keeps_first_valid_frame(tmp_path: Path):
    observations = [
        ObjectObservation(3, "yellow_object_0", [0, 0, 10, 10], 100, "detected", crop_path=tmp_path / "3.png"),
        ObjectObservation(8, "yellow_object_0", [0, 0, 30, 30], 900, "detected", crop_path=tmp_path / "8.png"),
    ]
    assert select_key_frames(observations, max_frames=1) == [3]


@pytest.mark.parametrize("raw_response", ["model output without JSON", "model output {not valid json"])
def test_parse_semantic_response_rejects_missing_or_malformed_json(raw_response: str):
    parsed = parse_semantic_response(raw_response)
    assert parsed == {"status": "parse_failed", "raw_response": raw_response}


def test_parse_semantic_response_rejects_missing_required_keys():
    response = json.loads(_valid_semantic_response())
    del response["material"]
    parsed = parse_semantic_response(json.dumps(response))
    assert parsed == {"status": "parse_failed", "raw_response": json.dumps(response)}


def test_parse_semantic_response_rejects_nonnumeric_confidence():
    raw_response = _valid_semantic_response(confidence="uncertain")
    parsed = parse_semantic_response(raw_response)
    assert parsed == {"status": "parse_failed", "raw_response": raw_response}


def test_parse_semantic_response_rejects_json_list():
    raw_response = json.dumps(["box", "yellow"])
    parsed = parse_semantic_response(raw_response)
    assert parsed == {"status": "parse_failed", "raw_response": raw_response}


def test_parse_semantic_response_skips_invalid_brace_group():
    valid = _valid_semantic_response(category="box")
    parsed = parse_semantic_response(f"example {{not json}} result {valid}")
    assert parsed["category"] == "box"
    assert parsed["status"] == "ok"


def test_select_key_frames_fills_available_budget(tmp_path: Path):
    observations = [
        ObjectObservation(
            frame_idx,
            "yellow_object_0",
            [0, 0, 10, 10],
            100 if frame_idx != 4 else 900,
            "detected",
            crop_path=tmp_path / f"{frame_idx}.png",
        )
        for frame_idx in range(5)
    ]
    selected = select_key_frames(observations, max_frames=4)
    assert len(selected) == 4
    assert selected[0] == 0
    assert 4 in selected


def test_build_qwen_messages_includes_text_and_image_data_url(tmp_path: Path):
    image_path = tmp_path / "crop.png"
    image_bytes = b"fake png bytes"
    image_path.write_bytes(image_bytes)

    messages = build_qwen_messages([image_path])
    content = messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert any(item["type"] == "text" for item in content)
    image_item = next(item for item in content if item["type"] == "image_url")
    data_url = image_item["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == image_bytes


def test_call_qwen_semantics_requires_api_key_before_network(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("network should not be called without an API key")

    monkeypatch.setattr("roboego_hand_vis.objects.semantic_qwen.urllib.request.urlopen", fail_if_called)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY is not set"):
        call_qwen_semantics(crop_paths=[], model="qwen-test")
