from __future__ import annotations

import json

from roboego_hand_vis.post_annotation_actions.model_subtasks import (
    build_object_resolution_payload,
    build_compact_subtask_payload,
    build_dry_run_report,
    frame_chunks,
    merge_chunk_subtasks,
    parse_model_subtask_response,
    write_model_subtask_output,
)


def _stage1_payload() -> dict:
    return {
        "stage": "robolabel_stage1_actions_merged",
        "source_chunk_dir": "/tmp/stage1_action_resolved_chunks",
        "source_files": ["/tmp/stage1_action_resolved_chunks/chunk_0000.json"],
        "frame_count": 4,
        "action_frame_count": 2,
        "frames": [
            {
                "t": 0,
                "scene": "white table with a black rack",
                "activity": "wearer rests hands on table",
                "hand_visible": True,
                "hands_visible": {"left": True, "right": True},
                "task_objects": ["rack"],
                "frame_actions": [],
            },
            {
                "t": 1,
                "scene": "white table with a black rack",
                "activity": "wearer reaches for a rack",
                "hand_visible": True,
                "hands_visible": {"left": True, "right": True},
                "task_objects": ["rack"],
                "frame_actions": [
                    {
                        "action": "",
                        "original_action": "reach",
                        "object_name": "rack",
                        "hand_side": "unknown",
                        "confidence": 0.6,
                        "phase_action": "approach",
                        "skill": "",
                        "subject": "unknown",
                        "manipulated_object": "rack",
                        "object_description": "black metal rack",
                        "object_track_id": "rack_1",
                    }
                ],
            },
            {
                "t": 2,
                "scene": "white table with a black rack",
                "activity": "wearer adjusts a rack",
                "hand_visible": True,
                "hands_visible": {"left": True, "right": True},
                "task_objects": ["rack"],
                "frame_actions": [
                    {
                        "action": "manipulate",
                        "original_action": "operate",
                        "object_name": "rack",
                        "hand_side": "both",
                        "confidence": 0.9,
                        "phase_action": "rotate",
                        "skill": "manipulate",
                        "subject": "both_effectors",
                        "manipulated_object": "rack",
                        "object_description": "black metal rack",
                        "object_track_id": "rack_1",
                    }
                ],
            },
            {
                "t": 3,
                "scene": "white table with a black rack",
                "activity": "wearer adjusts a rack",
                "hand_visible": True,
                "hands_visible": {"left": True, "right": True},
                "task_objects": ["rack"],
                "frame_actions": [
                    {
                        "action": "manipulate",
                        "original_action": "operate",
                        "object_name": "rack",
                        "hand_side": "both",
                        "confidence": 0.9,
                        "phase_action": "rotate",
                        "skill": "manipulate",
                        "subject": "both_effectors",
                        "manipulated_object": "rack",
                        "object_description": "black metal rack",
                        "object_track_id": "rack_1",
                    }
                ],
            },
        ],
    }


def test_build_compact_subtask_payload_keeps_model_context_without_extra_fields() -> None:
    payload = build_compact_subtask_payload(_stage1_payload())

    assert payload["task"] == "group_resolved_stage1_actions_into_subtasks"
    assert payload["source_stage"] == "robolabel_stage1_actions_merged"
    assert payload["frames"][0] == {
        "t": 0,
        "scene": "white table with a black rack",
        "activity": "wearer rests hands on table",
    }
    assert payload["frames"][1] == {
        "t": 1,
        "scene": "white table with a black rack",
        "activity": "wearer reaches for a rack",
    }


def test_compact_payload_keeps_scene_activity_and_normalized_verbs() -> None:
    payload = build_compact_subtask_payload(_stage1_payload())

    assert payload["frames"][2] == {
        "t": 2,
        "scene": "white table with a black rack",
        "activity": "wearer adjusts a rack",
    }


def test_old_frame_actions_are_not_sent_to_model() -> None:
    payload = build_compact_subtask_payload(_stage1_payload())

    assert "frame_actions" not in payload["frames"][1]
    assert "actions" not in payload["frames"][1]
    assert "objects" not in payload["frames"][1]
    assert set(payload["frames"][1]) == {"t", "scene", "activity"}


def test_compact_payload_uses_normalized_scene_for_action_frames() -> None:
    payload = build_compact_subtask_payload(_stage1_payload())

    assert payload["frames"][1]["scene"] == "white table with a black rack"
    assert payload["frames"][1]["activity"] == "wearer reaches for a rack"
    assert "actions" not in payload["frames"][1]
    assert "actions" not in payload["frames"][2]


def test_parse_model_subtask_response_validates_action_vocabulary() -> None:
    response = json.dumps(
        {
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 0,
                    "end_ts": 0,
                    "start_frame": 0,
                    "end_frame": 0,
                    "scene": "white table with a black rack",
                    "hand_visible": True,
                    "hands_visible": {"left": True, "right": True},
                    "task_objects": ["rack"],
                    "action": "idle",
                    "description": "wearer rests hands on table",
                    "object_name": None,
                    "hand_side": "both",
                    "confidence": 0.95,
                    "phase_action": "idle",
                    "skill": "idle",
                    "subject": "both_effectors",
                    "manipulated_object": None,
                    "object_description": None,
                    "source_frames": [0],
                    "actions": [],
                }
            ]
        }
    )

    parsed = parse_model_subtask_response(response)

    assert parsed["status"] == "ok"
    assert parsed["subtasks"][0]["action"] == "idle"


def test_parse_model_subtask_response_accepts_skill_level_place_action() -> None:
    response = json.dumps(
        {
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 20,
                    "end_ts": 20,
                    "start_frame": 20,
                    "end_frame": 20,
                    "action": "place",
                    "description": "wearer place cloth",
                    "object_name": "cloth",
                    "hand_side": "both",
                    "confidence": 0.9,
                    "phase_action": "release",
                    "skill": "place",
                    "source_frames": [20],
                    "actions": [],
                }
            ]
        }
    )

    parsed = parse_model_subtask_response(response)

    assert parsed["status"] == "ok"
    assert parsed["subtasks"][0]["action"] == "place"
    assert parsed["subtasks"][0]["phase_action"] == "release"


def test_parse_model_subtask_response_accepts_empty_skill_with_phase_action() -> None:
    response = json.dumps(
        {
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 1,
                    "end_ts": 1,
                    "start_frame": 1,
                    "end_frame": 1,
                    "action": "approach",
                    "description": "wearer approach rack",
                    "object_name": None,
                    "hand_side": "unknown",
                    "confidence": 0.6,
                    "phase_action": "approach",
                    "skill": "",
                    "source_frames": [1],
                    "actions": [
                        {
                            "t": 1,
                            "action": "",
                            "phase_action": "approach",
                            "skill": "",
                            "confidence": 0.6,
                        }
                    ],
                }
            ]
        }
    )

    parsed = parse_model_subtask_response(response)

    assert parsed["status"] == "ok"
    assert parsed["subtasks"][0]["action"] == "approach"
    assert "skill" not in parsed["subtasks"][0]


def test_parse_model_subtask_response_normalizes_phase_name_misused_as_skill() -> None:
    response = json.dumps(
        {
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 7,
                    "end_ts": 8,
                    "start_frame": 7,
                    "end_frame": 8,
                    "action": "align",
                    "description": "wearer align rack",
                    "object_name": None,
                    "hand_side": "unknown",
                    "confidence": 0.7,
                    "phase_action": "align",
                    "skill": "align",
                    "source_frames": [7, 8],
                    "actions": [
                        {
                            "t": 7,
                            "action": "",
                            "phase_action": "align",
                            "skill": "align",
                            "confidence": 0.7,
                        }
                    ],
                }
            ]
        }
    )

    parsed = parse_model_subtask_response(response)

    assert parsed["status"] == "ok"
    assert parsed["subtasks"][0]["action"] == "align"
    assert parsed["subtasks"][0]["phase_action"] == "align"
    assert "skill" not in parsed["subtasks"][0]


def test_parse_model_subtask_response_accepts_selected_original_action() -> None:
    response = json.dumps(
        {
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 14,
                    "end_ts": 16,
                    "start_frame": 14,
                    "end_frame": 16,
                    "selected_action": "unfold",
                    "description": "wearer fold pouch",
                    "source_frames": [14, 15, 16],
                    "actions": [],
                }
            ]
        }
    )

    parsed = parse_model_subtask_response(response)

    assert parsed["status"] == "ok"
    assert parsed["subtasks"][0]["action"] == "fold"
    assert parsed["subtasks"][0]["skill"] == "fold"
    assert parsed["subtasks"][0]["phase_action"] == "fold_motion"


def test_write_model_subtask_output_enriches_selected_action_from_source_frames(tmp_path) -> None:
    input_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    output_json = tmp_path / "stage1_subtasks_left_v1.json"
    input_json.write_text(json.dumps(_stage1_payload()), encoding="utf-8")

    def fake_resolver(review_payload: dict) -> dict:
        assert review_payload["frames"][2] == {
            "t": 2,
            "scene": "white table with a black rack",
            "activity": "wearer adjusts a rack",
        }
        return {
            "status": "ok",
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 2,
                    "end_ts": 3,
                    "start_frame": 2,
                    "end_frame": 3,
                    "source_frames": [2, 3],
                }
            ],
        }

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=fake_resolver,
        chunk_size=0,
    )

    subtask = output["subtasks"][0]
    assert subtask["action"] == "manipulate"
    assert "skill" not in subtask
    assert subtask["phase_action"] == "rotate"
    assert subtask["description"] == "wearer manipulate rack"


def test_write_model_subtask_output_prunes_idle_default_fields(tmp_path) -> None:
    input_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    output_json = tmp_path / "stage1_subtasks_left_v1.json"
    input_json.write_text(json.dumps(_stage1_payload()), encoding="utf-8")

    def fake_resolver(review_payload: dict) -> dict:
        assert set(review_payload["frames"][0]) == {"t", "scene", "activity"}
        assert set(review_payload["frames"][1]) == {"t", "scene", "activity"}
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

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=fake_resolver,
        chunk_size=0,
    )

    subtask = output["subtasks"][0]
    assert subtask["action"] == "idle"
    assert subtask["description"] == "wearer rests hands on table"
    assert subtask["hand_visible"] is True
    assert subtask["hands_visible"] == {"left": True, "right": True}
    assert subtask["task_objects"] == ["rack"]
    for noisy_field in [
        "selected_action",
        "phase_action",
        "skill",
        "object_name",
        "hand_side",
        "confidence",
        "subject",
        "manipulated_object",
        "object_description",
        "object_aliases",
        "actions",
        "source_chunk_index",
    ]:
        assert noisy_field not in subtask


def test_write_model_subtask_output_uses_model_object_aliases_for_canonical_name(tmp_path) -> None:
    payload = _stage1_payload()
    payload["frames"][2]["frame_actions"][0]["object_name"] = "rack"
    payload["frames"][3]["frame_actions"][0]["object_name"] = "shelf"
    input_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    output_json = tmp_path / "stage1_subtasks_left_v1.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    def fake_resolver(review_payload: dict) -> dict:
        return {
            "status": "ok",
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 2,
                    "end_ts": 3,
                    "start_frame": 2,
                    "end_frame": 3,
                    "source_frames": [2, 3],
                    "object_aliases": [
                        {
                            "canonical_name": "rack",
                            "aliases": ["rack", "shelf", "black shelf unit"],
                        }
                    ],
                }
            ],
        }

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=fake_resolver,
        chunk_size=0,
    )

    subtask = output["subtasks"][0]
    assert subtask["object_name"] == "rack"
    assert subtask["description"] == "wearer manipulate rack"
    assert "object_aliases" not in subtask


def test_write_model_subtask_output_does_not_use_unrelated_first_alias_group(tmp_path) -> None:
    payload = _stage1_payload()
    payload["frames"][2]["frame_actions"][0].update(
        {
            "action": "pick",
            "skill": "pick",
            "phase_action": "grasp",
            "object_name": "bottle",
            "manipulated_object": "bottle",
            "object_description": "small brown bottle",
        }
    )
    input_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    output_json = tmp_path / "stage1_subtasks_left_v1.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    def fake_resolver(review_payload: dict) -> dict:
        return {
            "status": "ok",
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 2,
                    "end_ts": 2,
                    "start_frame": 2,
                    "end_frame": 2,
                    "source_frames": [2],
                    "object_aliases": [
                        {"canonical_name": "shelf", "aliases": ["shelf", "rack", "lower shelf"]},
                        {"canonical_name": "bottle", "aliases": ["bottle", "brown bottle"]},
                    ],
                }
            ],
        }

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=fake_resolver,
        chunk_size=0,
    )

    subtask = output["subtasks"][0]
    assert subtask["object_name"] == "bottle"
    assert subtask["description"] == "wearer pick bottle"


def test_write_model_subtask_output_prefers_task_action_over_hold(tmp_path) -> None:
    payload = _stage1_payload()
    payload["frames"][2]["frame_actions"] = [
        {
            "action": "hold",
            "skill": "hold",
            "phase_action": "secure",
            "object_name": "shelf",
            "hand_side": "left",
            "confidence": 0.8,
        },
        {
            "action": "wipe",
            "skill": "wipe",
            "phase_action": "scrub_motion",
            "object_name": "shelf",
            "hand_side": "right",
            "confidence": 0.9,
        },
    ]
    input_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    output_json = tmp_path / "stage1_subtasks_left_v1.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    def fake_resolver(review_payload: dict) -> dict:
        return {
            "status": "ok",
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 2,
                    "end_ts": 2,
                    "start_frame": 2,
                    "end_frame": 2,
                    "source_frames": [2],
                }
            ],
        }

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=fake_resolver,
        chunk_size=0,
    )

    subtask = output["subtasks"][0]
    assert subtask["action"] == "wipe"
    assert "skill" not in subtask
    assert subtask["phase_action"] == "scrub_motion"
    assert "phases" not in subtask
    assert subtask["description"] == "wearer wipe shelf"


def test_build_object_resolution_payload_groups_objects_per_subtask_interval() -> None:
    stage1 = _stage1_payload()
    stage1["frames"][2]["frame_actions"][0]["object_name"] = "rack"
    stage1["frames"][2]["frame_actions"][0]["object_description"] = "black metal two-tier rack"
    stage1["frames"][3]["frame_actions"][0]["object_name"] = "shelf"
    stage1["frames"][3]["frame_actions"][0]["object_description"] = "black two-tier metal shelf"
    subtasks = [
        {
            "id": 1,
            "start_ts": 2,
            "end_ts": 3,
            "start_frame": 2,
            "end_frame": 3,
            "source_frames": [2, 3],
        }
    ]

    payload = build_object_resolution_payload(stage1, subtasks)

    assert payload == {
        "task": "resolve_subtask_object_aliases",
        "source_stage": "robolabel_stage1_actions_merged",
        "subtasks": [
            {
                "id": 1,
                "source_frames": [2, 3],
                "activity_span": [
                    {"t": 2, "activity": "wearer adjusts a rack"},
                    {"t": 3, "activity": "wearer adjusts a rack"},
                ],
                "objects": [
                    {
                        "name": "rack",
                        "descriptions": ["black metal two-tier rack"],
                    },
                    {
                        "name": "shelf",
                        "descriptions": ["black two-tier metal shelf"],
                    },
                ],
            }
        ],
    }


def test_write_model_subtask_output_resolves_object_aliases_after_segmentation(tmp_path) -> None:
    payload = _stage1_payload()
    payload["frames"][2]["frame_actions"][0]["object_name"] = "rack"
    payload["frames"][2]["frame_actions"][0]["object_description"] = "black metal two-tier rack"
    payload["frames"][3]["frame_actions"][0]["object_name"] = "shelf"
    payload["frames"][3]["frame_actions"][0]["object_description"] = "black two-tier metal shelf"
    input_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    output_json = tmp_path / "stage1_subtasks_left_v1.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    def fake_segment_resolver(review_payload: dict) -> dict:
        assert set(review_payload["frames"][2]) == {"t", "scene", "activity"}
        return {
            "status": "ok",
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 2,
                    "end_ts": 3,
                    "start_frame": 2,
                    "end_frame": 3,
                    "source_frames": [2, 3],
                }
            ],
        }

    def fake_object_resolver(review_payload: dict) -> dict:
        assert review_payload["subtasks"][0]["objects"] == [
            {"name": "rack", "descriptions": ["black metal two-tier rack"]},
            {"name": "shelf", "descriptions": ["black two-tier metal shelf"]},
        ]
        return {
            "status": "ok",
            "object_aliases_by_subtask": {
                "1": [
                    {
                        "canonical_name": "shelf",
                        "aliases": ["shelf", "rack", "black metal two-tier rack"],
                    }
                ]
            },
        }

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=fake_segment_resolver,
        object_resolver=fake_object_resolver,
        chunk_size=0,
    )

    subtask = output["subtasks"][0]
    assert subtask["object_name"] == "shelf"
    assert "object_aliases" not in subtask
    assert subtask["description"] == "wearer manipulate shelf"


def test_write_model_subtask_output_includes_object_attributes_for_interval(tmp_path) -> None:
    input_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    output_json = tmp_path / "stage1_subtasks_left_v1.json"
    input_json.write_text(json.dumps(_stage1_payload()), encoding="utf-8")

    def fake_resolver(review_payload: dict) -> dict:
        return {
            "status": "ok",
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 2,
                    "end_ts": 3,
                    "start_frame": 2,
                    "end_frame": 3,
                    "source_frames": [2, 3],
                }
            ],
        }

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=fake_resolver,
        chunk_size=0,
    )

    subtask = output["subtasks"][0]
    assert subtask["objects"] == [
        {
            "name": "rack",
            "color": "black",
            "material": "metal",
            "shape": "rack",
            "description": "black metal rack",
        }
    ]


def test_write_model_subtask_output_preserves_top_level_metadata(tmp_path) -> None:
    input_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    output_json = tmp_path / "stage1_subtasks_left_v1.json"
    input_json.write_text(json.dumps(_stage1_payload()), encoding="utf-8")

    def fake_resolver(review_payload: dict) -> dict:
        assert review_payload["frames"][0]["scene"] == "white table with a black rack"
        assert review_payload["frames"][0]["activity"] == "wearer rests hands on table"
        return {
            "status": "ok",
            "subtasks": [
                {
                    "id": 1,
                    "start_ts": 0,
                    "end_ts": 0,
                    "start_frame": 0,
                    "end_frame": 0,
                    "scene": "white table with a black rack",
                    "hand_visible": True,
                    "hands_visible": {"left": True, "right": True},
                    "task_objects": ["rack"],
                    "action": "idle",
                    "description": "wearer rests hands on table",
                    "object_name": None,
                    "hand_side": "both",
                    "confidence": 0.95,
                    "phase_action": "idle",
                    "skill": "idle",
                    "subject": "both_effectors",
                    "manipulated_object": None,
                    "object_description": None,
                    "source_frames": [0],
                    "actions": [],
                },
                {
                    "id": 2,
                    "start_ts": 2,
                    "end_ts": 3,
                    "start_frame": 2,
                    "end_frame": 3,
                    "scene": "white table with a black rack",
                    "hand_visible": True,
                    "hands_visible": {"left": True, "right": True},
                    "task_objects": ["rack"],
                    "action": "manipulate",
                    "description": "wearer manipulate rack",
                    "object_name": "rack",
                    "hand_side": "both",
                    "confidence": 0.9,
                    "phase_action": "rotate",
                    "skill": "manipulate",
                    "subject": "both_effectors",
                    "manipulated_object": "rack",
                    "object_description": "black metal rack",
                    "source_frames": [2, 3],
                    "actions": [
                        {
                            "t": 2,
                            "action": "manipulate",
                            "object_name": "rack",
                            "hand_side": "both",
                            "confidence": 0.9,
                            "phase_action": "rotate",
                            "skill": "manipulate",
                            "subject": "both_effectors",
                            "manipulated_object": "rack",
                            "object_description": "black metal rack",
                            "object_track_id": "rack_1",
                        }
                    ],
                },
            ],
        }

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=fake_resolver,
        chunk_size=0,
    )

    assert output["stage"] == "robolabel_stage1_actions_merged"
    assert output["source_chunk_dir"] == "/tmp/stage1_action_resolved_chunks"
    assert output["frame_count"] == 4
    assert output["action_frame_count"] == 2
    assert output["subtask_stage"] == "robolabel_stage1_subtasks_from_model"
    assert output["subtask_count"] == 2
    assert output["subtasks"][1]["description"] == "wearer manipulate rack"
    assert json.loads(output_json.read_text(encoding="utf-8")) == output


def test_frame_chunks_support_overlap_for_main_runner() -> None:
    frames = [{"t": idx} for idx in range(5)]

    chunks = frame_chunks(frames, chunk_size=3, overlap=1)

    assert [[frame["t"] for frame in chunk] for chunk in chunks] == [[0, 1, 2], [2, 3, 4]]


def test_merge_chunk_subtasks_renumbers_and_deduplicates_overlap() -> None:
    merged = merge_chunk_subtasks(
        [
            {
                "chunk_index": 0,
                "subtasks": [
                    {"id": 1, "action": "idle", "description": "rest", "source_frames": [0]},
                    {"id": 2, "action": "approach", "description": "approach rack", "source_frames": [1]},
                ],
            },
            {
                "chunk_index": 1,
                "subtasks": [
                    {"id": 1, "action": "approach", "description": "approach rack", "source_frames": [1]},
                    {"id": 2, "action": "secure", "description": "secure rack", "source_frames": [2]},
                ],
            },
        ]
    )

    assert [subtask["id"] for subtask in merged] == [1, 2, 3]
    assert [subtask["description"] for subtask in merged] == ["rest", "approach rack", "secure rack"]


def test_merge_chunk_subtasks_trims_boundary_overlap_for_different_actions() -> None:
    merged = merge_chunk_subtasks(
        [
            {
                "chunk_index": 0,
                "subtasks": [
                    {
                        "id": 1,
                        "start_ts": 15,
                        "end_ts": 21,
                        "start_frame": 15,
                        "end_frame": 21,
                        "source_frames": [15, 16, 17, 18, 19, 20, 21],
                        "action": "hold",
                        "skill": "hold",
                        "object_name": "cloth",
                    }
                ],
            },
            {
                "chunk_index": 1,
                "subtasks": [
                    {
                        "id": 1,
                        "start_ts": 21,
                        "end_ts": 29,
                        "start_frame": 21,
                        "end_frame": 29,
                        "source_frames": [21, 22, 23, 24, 25, 26, 27, 28, 29],
                        "action": "wipe",
                        "skill": "wipe",
                        "object_name": "shelf",
                    }
                ],
            },
        ]
    )

    assert merged[0]["source_frames"] == [15, 16, 17, 18, 19, 20]
    assert merged[0]["end_ts"] == 20
    assert merged[0]["end_frame"] == 20
    assert merged[1]["source_frames"] == [21, 22, 23, 24, 25, 26, 27, 28, 29]


def test_merge_chunk_subtasks_coalesces_adjacent_same_action_segments() -> None:
    merged = merge_chunk_subtasks(
        [
            {
                "chunk_index": 0,
                "subtasks": [
                    {
                        "id": 1,
                        "start_ts": 14,
                        "end_ts": 16,
                        "start_frame": 14,
                        "end_frame": 16,
                        "selected_action": "fold",
                        "action": "fold",
                        "skill": "fold",
                        "phase_action": "fold_motion",
                        "description": "wearer fold pouch",
                        "source_frames": [14, 15, 16],
                    },
                    {
                        "id": 2,
                        "start_ts": 17,
                        "end_ts": 21,
                        "start_frame": 17,
                        "end_frame": 21,
                        "selected_action": "fold",
                        "action": "fold",
                        "skill": "fold",
                        "phase_action": "fold_motion",
                        "description": "wearer fold pouch",
                        "source_frames": [17, 18, 19, 20, 21],
                    },
                ],
            }
        ]
    )

    assert len(merged) == 1
    assert merged[0]["id"] == 1
    assert merged[0]["start_ts"] == 14
    assert merged[0]["end_ts"] == 21
    assert merged[0]["start_frame"] == 14
    assert merged[0]["end_frame"] == 21
    assert merged[0]["source_frames"] == [14, 15, 16, 17, 18, 19, 20, 21]


def test_merge_chunk_subtasks_coalesces_same_skill_across_phase_changes() -> None:
    merged = merge_chunk_subtasks(
        [
            {
                "chunk_index": 0,
                "subtasks": [
                    {
                        "id": 1,
                        "start_ts": 3,
                        "end_ts": 7,
                        "start_frame": 3,
                        "end_frame": 7,
                        "action": "manipulate",
                        "skill": "manipulate",
                        "phase_action": "rotate",
                        "description": "wearer manipulate rack",
                        "object_name": "rack",
                        "source_frames": [3, 4, 5, 6, 7],
                    },
                    {
                        "id": 2,
                        "start_ts": 7,
                        "end_ts": 9,
                        "start_frame": 7,
                        "end_frame": 9,
                        "action": "manipulate",
                        "skill": "manipulate",
                        "phase_action": "align",
                        "description": "wearer manipulate shelf",
                        "object_name": "shelf",
                        "source_frames": [7, 8, 9],
                    },
                ],
            }
        ]
    )

    assert len(merged) == 1
    assert merged[0]["start_ts"] == 3
    assert merged[0]["end_ts"] == 9
    assert merged[0]["source_frames"] == [3, 4, 5, 6, 7, 8, 9]
    assert merged[0]["action"] == "manipulate"
    assert merged[0]["skill"] == "manipulate"
    assert merged[0]["phases"] == [
        {
            "phase_action": "rotate",
            "start_ts": 3,
            "end_ts": 7,
            "start_frame": 3,
            "end_frame": 7,
            "source_frames": [3, 4, 5, 6, 7],
        },
        {
            "phase_action": "align",
            "start_ts": 7,
            "end_ts": 9,
            "start_frame": 7,
            "end_frame": 9,
            "source_frames": [7, 8, 9],
        },
    ]


def test_write_model_subtask_output_can_call_resolver_per_chunk(tmp_path) -> None:
    input_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    output_json = tmp_path / "stage1_subtasks_left_v1.json"
    input_json.write_text(json.dumps(_stage1_payload()), encoding="utf-8")
    seen_chunks = []

    def fake_resolver(review_payload: dict) -> dict:
        seen_chunks.append([frame["t"] for frame in review_payload["frames"]])
        start_t = review_payload["frames"][0]["t"]
        return {
            "status": "ok",
            "subtasks": [
                {
                    "id": 99,
                    "start_ts": start_t,
                    "end_ts": start_t,
                    "start_frame": start_t,
                    "end_frame": start_t,
                    "scene": "white table with a black rack",
                    "hand_visible": True,
                    "hands_visible": {"left": True, "right": True},
                    "task_objects": ["rack"],
                    "action": "idle",
                    "description": f"chunk starts at {start_t}",
                    "object_name": None,
                    "hand_side": "both",
                    "confidence": 0.8,
                    "phase_action": "idle",
                    "skill": "idle",
                    "subject": "both_effectors",
                    "manipulated_object": None,
                    "object_description": None,
                    "source_frames": [start_t],
                    "actions": [],
                }
            ],
        }

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=fake_resolver,
        chunk_size=2,
        overlap=0,
    )

    assert seen_chunks == [[0, 1], [2, 3]]
    assert output["subtask_stage"] == "robolabel_stage1_subtasks_from_model_chunked"
    assert output["chunk_count"] == 2
    assert output["subtask_count"] == 2


def test_build_dry_run_report_shows_chunk_sizes() -> None:
    report = build_dry_run_report(_stage1_payload(), chunk_size=2, overlap=0)

    assert report["stage"] == "dry_run_stage1_subtasks_from_model_chunked"
    assert report["chunk_count"] == 2
    assert report["chunks"][0]["start_t"] == 0
    assert report["chunks"][0]["end_t"] == 1
    assert report["chunks"][0]["frame_count"] == 2
    assert report["chunks"][0]["compact_bytes"] > 0
