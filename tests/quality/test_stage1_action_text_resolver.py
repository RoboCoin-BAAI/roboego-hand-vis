import json
from pathlib import Path

from roboego_hand_vis.quality.stage1_action_text_resolver import (
    build_text_resolver_payload,
    build_text_resolver_messages,
    build_text_resolver_prompt,
    find_uncertain_action_frames,
    merge_text_resolver_results,
    parse_text_resolver_response,
    write_text_resolver_review_input,
)


def frame_with_actions() -> dict:
    return {
        "t": 151,
        "activity": "wearer places the phone on the desk and uses the right hand to move the computer mouse",
        "hands_visible": {"left": True, "right": True},
        "interaction_object_attributes": [
            {"name": "phone", "hand_side": "left"},
            {"name": "mouse", "hand_side": "right"},
        ],
        "frame_actions": [
            {"action": "place", "object_name": "phone", "hand_side": "left", "source": "activity_object_match", "confidence": 0.9},
            {"action": "operate", "object_name": "unknown", "hand_side": "right", "source": "unbound", "confidence": 0.45},
        ],
    }


def test_find_uncertain_action_frames_selects_unbound_and_low_confidence():
    payload = {"frames": [frame_with_actions(), {"t": 152, "frame_actions": []}]}

    review_frames = find_uncertain_action_frames(payload)

    assert [frame["t"] for frame in review_frames] == [151]


def test_build_text_resolver_payload_keeps_only_compact_text_fields():
    review = build_text_resolver_payload([frame_with_actions()])

    assert review["allowed_actions"]
    assert review["frames"] == [
        {
            "t": 151,
            "activity": "wearer places the phone on the desk and uses the right hand to move the computer mouse",
            "hands_visible": {"left": True, "right": True},
            "objects": [
                {"name": "phone", "hand_side": "left"},
                {"name": "mouse", "hand_side": "right"},
            ],
            "current_frame_actions": [
                {"action": "place", "object_name": "phone", "hand_side": "left", "source": "activity_object_match", "confidence": 0.9},
                {"action": "operate", "object_name": "unknown", "hand_side": "right", "source": "unbound", "confidence": 0.45},
            ],
        }
    ]


def test_build_text_resolver_messages_contains_strict_json_instruction():
    review = build_text_resolver_payload([frame_with_actions()])

    messages = build_text_resolver_messages(review)

    assert messages[0]["role"] == "system"
    assert "strict JSON" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "resolve_stage1_hand_action_object_bindings" in messages[1]["content"]


def test_build_text_resolver_prompt_mentions_no_invented_objects():
    prompt = build_text_resolver_prompt(build_text_resolver_payload([frame_with_actions()]))

    assert "Do not invent object names" in prompt
    assert "wearer places the phone" in prompt


def test_parse_text_resolver_response_validates_actions():
    response = json.dumps(
        {
            "frames": [
                {
                    "t": 151,
                    "frame_actions": [
                        {"action": "place", "object_name": "phone", "hand_side": "left", "confidence": 0.9},
                        {"action": "operate", "object_name": "mouse", "hand_side": "right", "confidence": 0.85},
                    ],
                }
            ]
        }
    )

    parsed = parse_text_resolver_response(response)

    assert parsed["status"] == "ok"
    assert parsed["frames"][0]["frame_actions"][1]["source"] == "qwen_text_resolver"


def test_merge_text_resolver_results_replaces_uncertain_actions_only():
    payload = {"frames": [frame_with_actions()]}
    resolved = {
        "frames": [
            {
                "t": 151,
                "frame_actions": [
                    {"action": "place", "object_name": "phone", "hand_side": "left", "source": "qwen_text_resolver", "confidence": 0.9},
                    {"action": "operate", "object_name": "mouse", "hand_side": "right", "source": "qwen_text_resolver", "confidence": 0.85},
                ],
            }
        ]
    }

    merged = merge_text_resolver_results(payload, resolved)

    assert merged["frames"][0]["frame_actions"][1]["object_name"] == "mouse"
    assert merged["frames"][0]["frame_actions"][1]["source"] == "qwen_text_resolver"


def test_write_text_resolver_review_input(tmp_path: Path):
    input_json = tmp_path / "actions.json"
    output_json = tmp_path / "review.json"
    input_json.write_text(json.dumps({"frames": [frame_with_actions()]}), encoding="utf-8")

    payload = write_text_resolver_review_input(input_json=input_json, output_json=output_json)

    assert json.loads(output_json.read_text(encoding="utf-8")) == payload
    assert payload["frame_count"] == 1
