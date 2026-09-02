import json
from pathlib import Path

from roboego_hand_vis.quality.hand_windows import HandWindowConfig, build_hand_estimation_windows, write_hand_estimation_windows


def frame(t: int, *, left: bool, right: bool, action: str | None = None, obj: str = "product") -> dict:
    actions = []
    if action:
        actions.append({"action": action, "object_name": obj, "hand_side": "unknown", "confidence": 0.9})
    return {
        "t": t,
        "hand_visible": left or right,
        "hands_visible": {"left": left, "right": right},
        "hand_object_interaction": bool(action),
        "interaction_objects": [obj] if action else [],
        "frame_actions": actions,
        "activity": f"frame {t}",
    }


def test_build_hand_windows_keeps_only_seconds_with_both_left_and_right_hands_visible():
    stage1 = {
        "frames": [
            frame(5, left=True, right=False, action="hold"),
            frame(6, left=True, right=True, action="touch"),
            frame(7, left=True, right=True, action="place"),
            frame(8, left=False, right=True, action="point"),
            frame(10, left=True, right=True, action="operate", obj="phone"),
        ]
    }
    plan = {
        "segments": [
            {"segment_id": "seg_0000", "start": 5, "end": 10, "semantic_label": "touch_place_operate"}
        ]
    }

    windows = build_hand_estimation_windows(
        stage1,
        plan,
        config=HandWindowConfig(context_expand_s=0, merge_gap_s=0, min_window_s=1),
    )

    assert [window["start"] for window in windows["windows"]] == [6, 10]
    assert [window["end"] for window in windows["windows"]] == [7, 10]
    assert windows["windows"][0]["hands_required"] == ["left", "right"]
    assert windows["windows"][0]["segment_frame_range"] == {"start": 30, "end": 89, "num_frames": 60}
    assert windows["windows"][0]["actions"] == ["place", "touch"]
    assert windows["windows"][1]["objects"] == ["phone"]


def test_build_hand_windows_uses_window_actions_for_semantic_label_and_keeps_source_label():
    stage1 = {
        "frames": [
            frame(10, left=True, right=True, action="touch"),
            frame(11, left=True, right=True, action="place"),
        ]
    }
    plan = {
        "segments": [
            {
                "segment_id": "seg_0000",
                "start": 0,
                "end": 20,
                "semantic_label": "lift_open_operate_pick",
            }
        ]
    }

    windows = build_hand_estimation_windows(
        stage1,
        plan,
        config=HandWindowConfig(context_expand_s=0, merge_gap_s=0, min_window_s=1),
    )

    window = windows["windows"][0]
    assert window["actions"] == ["place", "touch"]
    assert window["semantic_label"] == "place_touch"
    assert window["source_segment_semantic_label"] == "lift_open_operate_pick"


def test_build_hand_windows_expands_and_merges_close_two_hand_action_seconds():
    stage1 = {
        "frames": [
            frame(10, left=True, right=True, action="touch"),
            frame(11, left=True, right=False),
            frame(12, left=True, right=True, action="operate"),
        ]
    }
    plan = {"segments": [{"segment_id": "seg_0000", "start": 10, "end": 12, "semantic_label": "touch_operate"}]}

    windows = build_hand_estimation_windows(
        stage1,
        plan,
        config=HandWindowConfig(context_expand_s=1, merge_gap_s=1, min_window_s=1),
    )

    assert len(windows["windows"]) == 1
    assert windows["windows"][0]["start"] == 10
    assert windows["windows"][0]["end"] == 12
    assert windows["windows"][0]["core_range"] == {"start": 10, "end": 12}


def test_write_hand_estimation_windows_writes_json(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    plan = tmp_path / "plan.json"
    output = tmp_path / "hand_windows.json"
    stage1.write_text(json.dumps({"frames": [frame(1, left=True, right=True, action="touch")]}), encoding="utf-8")
    plan.write_text(json.dumps({"segments": [{"segment_id": "seg_0000", "start": 1, "end": 1}]}), encoding="utf-8")

    payload = write_hand_estimation_windows(
        stage1_actions_json=stage1,
        processing_plan_json=plan,
        output_json=output,
        config=HandWindowConfig(context_expand_s=0, min_window_s=1),
    )

    assert output.exists()
    assert payload["window_count"] == 1


def test_write_hand_estimation_windows_accepts_stage1_action_directory(tmp_path: Path):
    stage1_dir = tmp_path / "stage1_action_resolved_chunks"
    plan = tmp_path / "plan.json"
    output = tmp_path / "hand_windows.json"
    stage1_dir.mkdir()
    (stage1_dir / "chunk_0000.json").write_text(
        json.dumps({"frames": [frame(1, left=True, right=True, action="touch")]}),
        encoding="utf-8",
    )
    (stage1_dir / "chunk_0001.json").write_text(
        json.dumps({"frames": [frame(5, left=True, right=True, action="operate")]}),
        encoding="utf-8",
    )
    plan.write_text(
        json.dumps({"segments": [{"segment_id": "seg_0000", "start": 1, "end": 5}]}),
        encoding="utf-8",
    )

    payload = write_hand_estimation_windows(
        stage1_actions_json=stage1_dir,
        processing_plan_json=plan,
        output_json=output,
        config=HandWindowConfig(context_expand_s=0, merge_gap_s=0, min_window_s=1),
    )

    assert output.exists()
    assert payload["source_stage1_actions"] == str(stage1_dir)
    assert payload["window_count"] == 2


def test_build_hand_windows_can_split_single_and_two_hand_policy_windows():
    stage1 = {
        "frames": [
            frame(7, left=False, right=True, action="point"),
            frame(8, left=False, right=True, action="touch"),
            frame(11, left=True, right=True, action="hold"),
            frame(12, left=True, right=True, action="operate", obj="phone"),
            frame(14, left=True, right=False, action="hold", obj="phone"),
        ]
    }
    plan = {"segments": [{"segment_id": "seg_0000", "start": 5, "end": 15, "semantic_label": "operate"}]}

    windows = build_hand_estimation_windows(
        stage1,
        plan,
        config=HandWindowConfig(context_expand_s=0, merge_gap_s=0, hand_window_mode="single_and_two"),
    )

    by_type = {(window["window_type"], tuple(window["hands_required"])): window for window in windows["windows"]}
    assert by_type[("single_hand", ("right",))]["recommended_stage1_hand_policy"] == "conservative"
    assert by_type[("two_hand", ("left", "right"))]["recommended_stage1_hand_policy"] == "permissive"
    assert by_type[("single_hand", ("left",))]["recommended_stage1_hand_policy"] == "conservative"
    assert by_type[("single_hand", ("right",))]["core_range"] == {"start": 7, "end": 8}
    assert by_type[("two_hand", ("left", "right"))]["core_range"] == {"start": 11, "end": 12}


def test_build_hand_windows_can_merge_adjacent_windows_after_context_expand():
    stage1 = {
        "frames": [
            frame(10, left=True, right=True, action="touch"),
            frame(15, left=True, right=True, action="place"),
        ]
    }
    plan = {"segments": [{"segment_id": "seg_0000", "start": 8, "end": 17, "semantic_label": "touch_place"}]}

    windows = build_hand_estimation_windows(
        stage1,
        plan,
        config=HandWindowConfig(context_expand_s=2, merge_gap_s=1, merge_after_expand=True),
    )

    assert len(windows["windows"]) == 1
    assert windows["windows"][0]["start"] == 8
    assert windows["windows"][0]["end"] == 17
    assert windows["windows"][0]["core_range"] == {"start": 10, "end": 15}


def test_build_hand_windows_filters_by_min_priority_and_core_duration():
    stage1 = {
        "frames": [
            frame(1, left=True, right=True, action="operate"),
            frame(2, left=True, right=True, action="operate"),
            frame(5, left=True, right=True, action="touch"),
            frame(6, left=True, right=True, action="touch"),
            frame(7, left=True, right=True, action="touch"),
            frame(8, left=True, right=True, action="touch"),
            frame(9, left=True, right=True, action="touch"),
            frame(15, left=True, right=True, action="hold"),
            frame(16, left=True, right=True, action="hold"),
            frame(17, left=True, right=True, action="hold"),
            frame(18, left=True, right=True, action="hold"),
            frame(19, left=True, right=True, action="hold"),
        ]
    }
    plan = {"segments": [{"segment_id": "seg_0000", "start": 0, "end": 20, "semantic_label": "operate_touch_hold"}]}

    windows = build_hand_estimation_windows(
        stage1,
        plan,
        config=HandWindowConfig(
            context_expand_s=0,
            merge_gap_s=0,
            min_priority="high",
            min_core_duration_s=5,
        ),
    )

    assert len(windows["windows"]) == 1
    assert windows["windows"][0]["core_range"] == {"start": 5, "end": 9}
    assert windows["windows"][0]["priority"] == "high"
