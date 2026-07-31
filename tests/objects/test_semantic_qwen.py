from pathlib import Path

from roboego_hand_vis.objects.semantic_qwen import parse_semantic_response, select_key_frames
from roboego_hand_vis.objects.segmentation import ObjectObservation


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
