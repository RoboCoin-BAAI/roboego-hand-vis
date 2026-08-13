import json
from pathlib import Path

from roboego_hand_vis.quality.processing_plan import (
    ProcessingPlanConfig,
    build_processing_plan,
    build_segment_summaries,
    write_processing_plan,
    write_segment_summaries,
)


def frame(t: int, activity: str, *, hand: bool = True, action: str | None = None) -> dict:
    out = {
        "t": t,
        "scene": "store aisle",
        "activity": activity,
        "hand_visible": hand,
        "hands_visible": {"left": hand, "right": False},
        "hand_object_interaction": hand,
        "interaction_objects": ["phone"] if hand else [],
        "interaction_object_attributes": [
            {
                "name": "phone",
                "hand_side": "left",
                "color": "black",
                "shape": "rectangular",
                "material": "plastic and glass",
                "visual_description": "black phone",
                "confidence": 0.9,
            }
        ]
        if hand
        else [],
        "attention_objects": [],
        "task_context": hand,
        "task_objects": ["phone"] if hand else [],
        "privacy": False,
        "privacy_types": [],
        "image_quality": "clear",
        "uncertainty": 0.1,
        "notes": "",
    }
    if action:
        out["frame_actions"] = [{"action": action, "object_name": "phone", "hand_side": "left", "source": "test", "confidence": 0.9}]
    return out


def test_build_processing_plan_marks_long_hold_and_walk_as_interruptions():
    frames = [
        *[frame(t, "wearer holds a phone in the left hand", action="hold") for t in range(0, 8)],
        frame(8, "wearer picks up a product", action="pick"),
        *[frame(t, "wearer walks while holding a phone", action="hold") for t in range(9, 15)],
        frame(15, "wearer places a product into a shopping cart", action="place"),
    ]

    plan = build_processing_plan({"source_video": "left.mp4", "frames": frames}, config=ProcessingPlanConfig())

    interruptions = [unit for unit in plan["semantic_units"] if unit["unit_type"] == "interruption"]
    assert interruptions[0]["start"] == 0
    assert interruptions[0]["end"] == 7
    assert interruptions[0]["label"] == "long_hold_only"
    assert interruptions[1]["start"] == 9
    assert interruptions[1]["end"] == 14
    assert interruptions[1]["label"] == "walking_only"


def test_build_processing_plan_routes_long_hold_without_events_to_drop():
    frames = [frame(t, "wearer holds a phone in the left hand", action="hold") for t in range(0, 12)]

    plan = build_processing_plan({"source_video": "left.mp4", "frames": frames}, config=ProcessingPlanConfig())

    assert plan["drop_segments"] == [
        {
            "segment_id": "drop_0000",
            "start": 0,
            "end": 11,
            "duration": 12,
            "route": "drop",
            "semantic_label": "long_hold_only",
            "reason_tags": ["interruption", "long_hold_only", "no_event_core"],
            "needs_rectify": False,
            "needs_depth": False,
            "needs_hand_estimation": False,
            "needs_object_estimation": False,
        }
    ]


def test_build_processing_plan_keeps_only_edge_context_from_long_interruption():
    frames = [
        *[frame(t, "wearer holds a phone in the left hand", action="hold") for t in range(0, 12)],
        frame(12, "wearer picks up a product", action="pick"),
    ]

    plan = build_processing_plan(
        {"source_video": "left.mp4", "frames": frames},
        config=ProcessingPlanConfig(min_process_segment_s=1),
    )

    assert plan["segments"][0]["start"] == 9
    assert plan["segments"][0]["end"] == 12
    assert plan["drop_segments"][0]["start"] == 0
    assert plan["drop_segments"][0]["end"] == 8


def test_build_processing_plan_filters_short_process_segments_by_default():
    frames = [frame(0, "wearer picks up a product", action="pick")]

    plan = build_processing_plan({"source_video": "left.mp4", "frames": frames}, config=ProcessingPlanConfig())

    assert plan["segments"] == []


def test_write_processing_plan_merges_external_actions(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    actions = tmp_path / "actions.json"
    output = tmp_path / "plan.json"
    stage1.write_text(json.dumps({"source_video": "left.mp4", "frames": [frame(0, "wearer picks up a product")]}))
    actions.write_text(json.dumps({"frames": [{"t": 0, "frame_actions": [{"action": "pick", "object_name": "product"}]}]}))

    plan = write_processing_plan(
        stage1_json=stage1,
        output_json=output,
        actions_json=actions,
        config=ProcessingPlanConfig(min_process_segment_s=1),
    )

    assert output.exists()
    assert plan["semantic_units"][0]["unit_type"] == "event_core"


def test_build_segment_summaries_aggregates_actions_objects_and_privacy():
    frames = [
        frame(5, "wearer reaches for a product", action="reach"),
        frame(6, "wearer picks up a product", action="pick"),
        frame(7, "wearer places a product into a shopping cart", action="place"),
    ]
    frames[1]["privacy"] = True
    frames[1]["privacy_types"] = ["face"]
    for item in frames:
        item["interaction_objects"] = ["product"]
        item["interaction_object_attributes"][0]["name"] = "product"
        item["frame_actions"][0]["object_name"] = "product"
    plan = {
        "segments": [
            {
                "segment_id": "seg_0000",
                "start": 5,
                "end": 7,
                "duration": 3,
                "semantic_label": "pick_place",
            }
        ]
    }

    summaries = build_segment_summaries(plan, {"frames": frames})

    assert summaries["segments"][0]["segment_id"] == "seg_0000"
    assert summaries["segments"][0]["main_actions"] == ["pick", "place", "reach"]
    assert summaries["segments"][0]["main_objects"] == ["product"]
    assert summaries["segments"][0]["privacy"] is True
    assert summaries["segments"][0]["privacy_types"] == ["face"]
    assert "pick" in summaries["segments"][0]["summary"]


def test_write_segment_summaries_writes_total_and_per_segment_files(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "segment_summaries.json"
    segment_root = tmp_path / "segments"
    segment_dir = segment_root / "seg_0000_t000005_000007_pick_place"
    segment_dir.mkdir(parents=True)
    stage1.write_text(json.dumps({"frames": [frame(5, "wearer picks up a product", action="pick")]}))
    plan_path.write_text(json.dumps({"segments": [{"segment_id": "seg_0000", "start": 5, "end": 5, "duration": 1}]}))

    payload = write_segment_summaries(
        plan_json=plan_path,
        stage1_json=stage1,
        output_json=output,
        selected_segments_dir=segment_root,
    )

    assert output.exists()
    assert payload["segment_count"] == 1
    assert (segment_dir / "segment_summary.json").exists()
