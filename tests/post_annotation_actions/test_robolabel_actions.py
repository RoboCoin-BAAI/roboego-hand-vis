from __future__ import annotations

from roboego_hand_vis.post_annotation_actions.robolabel_actions import standardize_stage1_actions


def test_phase_only_action_leaves_skill_action_empty_and_omits_description() -> None:
    payload = {
        "stage": "stage1_action_extraction",
        "frames": [
            {
                "t": 3,
                "scene": "white table with a black rack, yellow pouch, and bottle",
                "activity": "wearer reaches for a rack",
                "interaction_object_attributes": [],
                "frame_actions": [
                    {
                        "action": "reach",
                        "object_name": "rack",
                        "hand_side": "unknown",
                        "confidence": 0.6,
                    }
                ],
            }
        ],
    }

    output = standardize_stage1_actions(payload)
    frame = output["frames"][0]
    action = frame["frame_actions"][0]

    assert frame["scene"] == "white table with a black rack, yellow pouch, and bottle"
    assert frame["activity"] == "wearer reaches for a rack"
    assert action["original_action"] == "reach"
    assert action["action"] == ""
    assert action["phase_action"] == "approach"
    assert action["skill"] == ""
    assert "description" not in action


def test_place_keeps_skill_level_action_and_uses_release_as_phase() -> None:
    payload = {
        "stage": "stage1_action_extraction",
        "frames": [
            {
                "t": 20,
                "scene": "white table with a shelf",
                "activity": "wearer places a cloth on a shelf",
                "interaction_object_attributes": [],
                "frame_actions": [
                    {
                        "action": "place",
                        "object_name": "cloth",
                        "hand_side": "both",
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    }

    output = standardize_stage1_actions(payload)
    action = output["frames"][0]["frame_actions"][0]

    assert action["original_action"] == "place"
    assert action["action"] == "place"
    assert action["skill"] == "place"
    assert action["phase_action"] == "release"
    assert action["description"] == "wearer place cloth"


def test_pick_keeps_skill_level_action_and_uses_grasp_as_phase() -> None:
    payload = {
        "stage": "stage1_action_extraction",
        "frames": [
            {
                "t": 44,
                "scene": "top-down view of a white desk with a black shelf unit",
                "activity": "wearer picks up a bottle",
                "interaction_object_attributes": [],
                "frame_actions": [
                    {
                        "action": "pick",
                        "object_name": "bottle",
                        "hand_side": "right",
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    }

    output = standardize_stage1_actions(payload)
    action = output["frames"][0]["frame_actions"][0]

    assert action["original_action"] == "pick"
    assert action["action"] == "pick"
    assert action["skill"] == "pick"
    assert action["phase_action"] == "grasp"
    assert action["description"] == "wearer pick bottle"


def test_wipe_uses_wipe_skill_and_scrub_motion_phase() -> None:
    payload = {
        "stage": "stage1_action_extraction",
        "frames": [
            {
                "t": 21,
                "scene": "top-down view of a shelf",
                "activity": "wearer wipes a shelf with a cloth",
                "interaction_object_attributes": [],
                "frame_actions": [
                    {
                        "action": "wipe",
                        "object_name": "shelf",
                        "hand_side": "both",
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    }

    output = standardize_stage1_actions(payload)
    action = output["frames"][0]["frame_actions"][0]

    assert action["original_action"] == "wipe"
    assert action["action"] == "wipe"
    assert action["skill"] == "wipe"
    assert action["phase_action"] == "scrub_motion"
    assert action["description"] == "wearer wipe shelf"
