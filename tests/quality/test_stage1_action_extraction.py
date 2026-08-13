import json
from pathlib import Path

from roboego_hand_vis.quality.stage1_actions import (
    extract_frame_actions,
    extract_stage1_actions,
    smooth_temporal_action_bindings,
    write_stage1_action_output,
)


def test_extract_frame_actions_binds_hold_and_push_to_named_objects():
    frame = {
        "t": 208,
        "activity": "wearer holds a phone and pushes a shopping cart through an aisle",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": True},
        "hand_object_interaction": True,
        "interaction_objects": ["phone", "shopping cart"],
        "interaction_object_attributes": [
            {"name": "phone", "hand_side": "left", "confidence": 0.95},
            {"name": "shopping cart", "hand_side": "right", "confidence": 0.85},
        ],
        "attention_objects": ["phone", "shopping cart"],
    }

    actions = extract_frame_actions(frame)

    assert actions == [
        {
            "action": "hold",
            "object_name": "phone",
            "hand_side": "left",
            "source": "activity_object_match",
            "confidence": 0.9,
        },
        {
            "action": "push",
            "object_name": "shopping cart",
            "hand_side": "right",
            "source": "activity_object_match",
            "confidence": 0.85,
        },
    ]


def test_extract_frame_actions_binds_single_object_when_activity_omits_name():
    frame = {
        "t": 10,
        "activity": "wearer picks up an item from the shelf",
        "hand_visible": True,
        "hands_visible": {"left": False, "right": True},
        "hand_object_interaction": True,
        "interaction_objects": ["product"],
        "interaction_object_attributes": [
            {"name": "product", "hand_side": "right", "confidence": 0.95},
        ],
        "attention_objects": ["product"],
    }

    actions = extract_frame_actions(frame)

    assert actions[0]["action"] == "pick"
    assert actions[0]["object_name"] == "product"
    assert actions[0]["hand_side"] == "right"
    assert actions[0]["source"] == "single_interaction_object"


def test_extract_frame_actions_binds_generic_terms_by_hand_side():
    frame = {
        "t": 12,
        "activity": "wearer holds a device in the left hand and pushes something with the right hand",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": True},
        "hand_object_interaction": True,
        "interaction_objects": ["phone", "shopping cart"],
        "interaction_object_attributes": [
            {"name": "phone", "hand_side": "left", "confidence": 0.95},
            {"name": "shopping cart", "hand_side": "right", "confidence": 0.9},
        ],
        "attention_objects": ["phone", "shopping cart"],
    }

    actions = extract_frame_actions(frame)

    assert actions == [
        {
            "action": "hold",
            "object_name": "phone",
            "hand_side": "left",
            "source": "hand_side_match",
            "confidence": 0.75,
        },
        {
            "action": "push",
            "object_name": "shopping cart",
            "hand_side": "right",
            "source": "hand_side_match",
            "confidence": 0.75,
        },
    ]


def test_extract_frame_actions_keeps_unknown_when_type_prior_has_multiple_candidates():
    frame = {
        "t": 13,
        "activity": "wearer scans something",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": False},
        "hand_object_interaction": True,
        "interaction_objects": ["product", "phone"],
        "interaction_object_attributes": [
            {"name": "product", "hand_side": "left", "confidence": 0.9},
            {"name": "phone", "hand_side": "left", "confidence": 0.9},
        ],
        "attention_objects": ["product", "phone"],
    }

    actions = extract_frame_actions(frame)

    assert actions == [
        {
            "action": "scan",
            "object_name": "unknown",
            "hand_side": "unknown",
            "source": "unbound",
            "confidence": 0.45,
        }
    ]


def test_extract_frame_actions_keeps_unknown_object_when_ambiguous():
    frame = {
        "t": 114,
        "activity": "wearer rotates something near the shelf",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": True},
        "hand_object_interaction": True,
        "interaction_objects": ["phone", "payment_terminal"],
        "interaction_object_attributes": [
            {"name": "phone", "hand_side": "left", "confidence": 0.9},
            {"name": "payment_terminal", "hand_side": "right", "confidence": 0.9},
        ],
        "attention_objects": ["phone", "payment_terminal"],
    }

    actions = extract_frame_actions(frame)

    assert actions == [
        {
            "action": "rotate",
            "object_name": "unknown",
            "hand_side": "unknown",
            "source": "unbound",
            "confidence": 0.45,
        }
    ]


def test_extract_frame_actions_does_not_bind_walking_to_held_object():
    frame = {
        "t": 142,
        "activity": "wearer walks while holding a device in the left hand and gesturing with the right hand",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": True},
        "hand_object_interaction": True,
        "interaction_objects": ["phone"],
        "interaction_object_attributes": [{"name": "phone", "hand_side": "left", "confidence": 0.95}],
        "attention_objects": ["phone"],
    }

    actions = extract_frame_actions(frame)

    assert actions == [
        {
            "action": "hold",
            "object_name": "phone",
            "hand_side": "left",
            "source": "single_interaction_object",
            "confidence": 0.8,
        }
    ]


def test_extract_frame_actions_extracts_state_and_attention_actions():
    frame = {
        "t": 20,
        "activity": "wearer reaches toward a product and points at it",
        "hand_visible": True,
        "hands_visible": {"left": False, "right": True},
        "hand_object_interaction": True,
        "interaction_objects": ["product"],
        "interaction_object_attributes": [{"name": "product", "hand_side": "right", "confidence": 0.9}],
        "attention_objects": ["product"],
    }

    actions = extract_frame_actions(frame)

    assert [action["action"] for action in actions] == ["reach", "point"]
    assert all(action["object_name"] == "product" for action in actions)


def test_extract_frame_actions_extracts_retail_and_container_actions():
    frame = {
        "t": 30,
        "activity": "wearer scans a product and puts it into a shopping cart",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": True},
        "hand_object_interaction": True,
        "interaction_objects": ["product", "shopping cart"],
        "interaction_object_attributes": [
            {"name": "product", "hand_side": "left", "confidence": 0.9},
            {"name": "shopping cart", "hand_side": "right", "confidence": 0.85},
        ],
        "attention_objects": ["product", "shopping cart"],
    }

    actions = extract_frame_actions(frame)

    assert [action["action"] for action in actions] == ["scan", "put into"]
    assert actions[0]["object_name"] == "product"
    assert actions[1]["object_name"] == "shopping cart"


def test_extract_frame_actions_extracts_operate_and_support():
    frame = {
        "t": 40,
        "activity": "wearer supports a box with the left hand and operates a terminal with the right hand",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": True},
        "hand_object_interaction": True,
        "interaction_objects": ["box", "terminal"],
        "interaction_object_attributes": [
            {"name": "box", "hand_side": "left", "confidence": 0.85},
            {"name": "terminal", "hand_side": "right", "confidence": 0.85},
        ],
        "attention_objects": ["terminal"],
    }

    actions = extract_frame_actions(frame)

    assert actions == [
        {
            "action": "support",
            "object_name": "box",
            "hand_side": "left",
            "source": "activity_object_match",
            "confidence": 0.85,
        },
        {
            "action": "operate",
            "object_name": "terminal",
            "hand_side": "right",
            "source": "activity_object_match",
            "confidence": 0.85,
        },
    ]


def test_extract_frame_actions_binds_use_hand_to_move_mouse_to_mouse():
    frame = {
        "t": 151,
        "activity": "wearer places the phone on the desk and uses the right hand to move the computer mouse",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": True},
        "hand_object_interaction": True,
        "interaction_objects": ["phone", "mouse"],
        "interaction_object_attributes": [
            {"name": "phone", "hand_side": "left", "confidence": 0.95},
            {"name": "mouse", "hand_side": "right", "confidence": 0.9},
        ],
        "attention_objects": ["phone", "mouse"],
    }

    actions = extract_frame_actions(frame)

    assert actions == [
        {
            "action": "place",
            "object_name": "phone",
            "hand_side": "left",
            "source": "activity_object_match",
            "confidence": 0.9,
        },
        {
            "action": "operate",
            "object_name": "mouse",
            "hand_side": "right",
            "source": "activity_object_match",
            "confidence": 0.9,
        },
    ]


def test_extract_frame_actions_ignores_non_hand_context_actions():
    frame = {
        "t": 50,
        "activity": "wearer looks at a product and appears to be paying at checkout",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": False},
        "hand_object_interaction": False,
        "interaction_objects": [],
        "interaction_object_attributes": [],
        "attention_objects": ["product"],
    }

    assert extract_frame_actions(frame) == []


def test_extract_frame_actions_uses_task_objects_when_interaction_attrs_are_missing():
    frame = {
        "t": 213,
        "activity": "wearer pushes a shopping cart while holding a phone",
        "hand_visible": True,
        "hands_visible": {"left": True, "right": False},
        "hand_object_interaction": False,
        "interaction_objects": [],
        "interaction_object_attributes": [],
        "attention_objects": ["shopping cart"],
        "task_objects": ["shopping cart", "phone"],
    }

    actions = extract_frame_actions(frame)

    assert actions == [
        {
            "action": "push",
            "object_name": "shopping cart",
            "hand_side": "unknown",
            "source": "task_object_match",
            "confidence": 0.6,
        },
        {
            "action": "hold",
            "object_name": "phone",
            "hand_side": "left",
            "source": "task_object_match",
            "confidence": 0.6,
        },
    ]


def test_extract_stage1_actions_preserves_only_frames_with_actions():
    payload = {
        "source_video": "left.mp4",
        "frames": [
            {
                "t": 0,
                "activity": "wearer stands near shelves",
                "hand_visible": False,
                "hands_visible": {"left": False, "right": False},
                "hand_object_interaction": False,
                "interaction_objects": [],
                "interaction_object_attributes": [],
                "attention_objects": [],
            },
            {
                "t": 1,
                "activity": "wearer presses a button",
                "hand_visible": True,
                "hands_visible": {"left": False, "right": True},
                "hand_object_interaction": True,
                "interaction_objects": ["button"],
                "interaction_object_attributes": [{"name": "button", "hand_side": "right", "confidence": 0.8}],
                "attention_objects": ["button"],
            },
        ],
    }

    output = extract_stage1_actions(payload)

    assert output["source_video"] == "left.mp4"
    assert output["frame_count"] == 2
    assert output["action_frame_count"] == 1
    assert output["frames"][1]["frame_actions"][0]["action"] == "press button"


def test_smooth_temporal_action_bindings_fills_unknown_between_same_hand_object():
    frames = [
        {
            "t": 0,
            "frame_actions": [{"action": "hold", "object_name": "phone", "hand_side": "left", "source": "activity_object_match", "confidence": 0.9}],
        },
        {
            "t": 1,
            "frame_actions": [{"action": "hold", "object_name": "unknown", "hand_side": "left", "source": "unbound", "confidence": 0.45}],
        },
        {
            "t": 2,
            "frame_actions": [{"action": "hold", "object_name": "phone", "hand_side": "left", "source": "activity_object_match", "confidence": 0.9}],
        },
    ]

    smoothed = smooth_temporal_action_bindings(frames)

    assert smoothed[1]["frame_actions"] == [
        {
            "action": "hold",
            "object_name": "phone",
            "hand_side": "left",
            "source": "temporal_carryover",
            "confidence": 0.7,
            "supporting_frames": [0, 2],
        }
    ]


def test_smooth_temporal_action_bindings_does_not_fill_when_neighbors_disagree():
    frames = [
        {
            "t": 0,
            "frame_actions": [{"action": "hold", "object_name": "phone", "hand_side": "left", "source": "activity_object_match", "confidence": 0.9}],
        },
        {
            "t": 1,
            "frame_actions": [{"action": "hold", "object_name": "unknown", "hand_side": "left", "source": "unbound", "confidence": 0.45}],
        },
        {
            "t": 2,
            "frame_actions": [{"action": "hold", "object_name": "product", "hand_side": "left", "source": "activity_object_match", "confidence": 0.9}],
        },
    ]

    smoothed = smooth_temporal_action_bindings(frames)

    assert smoothed[1]["frame_actions"][0]["object_name"] == "unknown"
    assert smoothed[1]["frame_actions"][0]["source"] == "unbound"


def test_write_stage1_action_output_reads_and_writes_json(tmp_path: Path):
    input_json = tmp_path / "stage1.json"
    output_json = tmp_path / "actions.json"
    input_json.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 1,
                        "activity": "wearer opens a door",
                        "hand_visible": True,
                        "hands_visible": {"left": True, "right": False},
                        "hand_object_interaction": True,
                        "interaction_objects": ["door"],
                        "interaction_object_attributes": [{"name": "door", "hand_side": "left", "confidence": 0.8}],
                        "attention_objects": ["door"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = write_stage1_action_output(input_json=input_json, output_json=output_json)

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved == payload
    assert saved["frames"][0]["frame_actions"][0]["action"] == "open"
