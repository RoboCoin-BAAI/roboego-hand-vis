import json
from pathlib import Path

from roboego_hand_vis.objects.sam3_hand_clip import (
    build_sam3_hand_clip_plan,
    build_stage1_prompt_text,
    build_sam3_hand_clip_plans_from_stage1,
    combine_sam3_object_tracks,
    collect_stage1_evidence,
    default_output_dir,
    infer_window_id_from_dir,
    infer_object_specs_from_stage1,
    materialize_interval_frames,
)


def test_collect_stage1_evidence_aligns_session_seconds_to_hand_clip_frames(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {"t": 146, "interaction_objects": ["shelf"]},
                    {
                        "t": 147,
                        "interaction_objects": ["phone"],
                        "interaction_object_attributes": ["black"],
                        "hands_visible": {"left": True, "right": True},
                    },
                    {
                        "t": 150,
                        "interaction_objects": ["mouse"],
                        "interaction_object_attributes": ["small", "black"],
                        "hands_visible": {"left": True, "right": True},
                    },
                    {"t": 162, "interaction_objects": ["cart"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "source_segment_range": {"start": 147, "end": 177},
        "segment_frame_range": {"start": 0, "end": 449, "num_frames": 450},
        "core_segment_frame_range": {"start": 0, "end": 389, "num_frames": 390},
        "start": 147,
        "end": 161,
    }

    evidence = collect_stage1_evidence(stage1, window, fps=30.0)

    assert [item["session_t"] for item in evidence] == [147, 150]
    assert evidence[0]["segment_frame"] == 0
    assert evidence[0]["clip_frame"] == 0
    assert evidence[1]["segment_frame"] == 90
    assert evidence[1]["clip_frame"] == 90


def test_build_sam3_hand_clip_plan_uses_frames_dir_and_writes_prompt_metadata(tmp_path: Path):
    hand_window = tmp_path / "handwin_0001_two_hand_left_right_000000_000449"
    frames_dir = hand_window / "frames"
    frames_dir.mkdir(parents=True)
    for idx in range(3):
        (frames_dir / f"frame_{idx:06d}.jpg").write_bytes(b"fake")
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 150,
                        "interaction_objects": ["mouse"],
                        "interaction_object_attributes": ["black"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "sam3_out"
    window = {
        "window_id": "handwin_0001",
        "source_segment_range": {"start": 147, "end": 177},
        "segment_frame_range": {"start": 0, "end": 449, "num_frames": 450},
        "core_segment_frame_range": {"start": 0, "end": 389, "num_frames": 390},
        "start": 147,
        "end": 161,
        "objects": ["mouse"],
    }

    plan = build_sam3_hand_clip_plan(
        hand_window_dir=hand_window,
        stage1_actions=stage1,
        window=window,
        object_name="mouse",
        object_shape="small rounded",
        object_color="black",
        output_dir=output_dir,
        sam3_script=Path("/sam3/run_sam3_video.py"),
        python_executable="/env/bin/python",
        fps=30.0,
    )

    assert plan.prompt_text == "black small rounded mouse"
    assert plan.prompt_frame_idx == 0
    assert plan.command[:4] == [
        "/env/bin/python",
        "/sam3/run_sam3_video.py",
        "--video",
        str(frames_dir),
    ]
    assert plan.command[plan.command.index("--text") + 1] == "black small rounded mouse"
    assert plan.command[plan.command.index("--obj_id") + 1] == "object_0"
    assert plan.command[plan.command.index("--output_dir") + 1] == str(output_dir)

    metadata = json.loads((output_dir / "sam3_prompt.json").read_text())
    assert metadata["prompt"]["text"] == "black small rounded mouse"
    assert metadata["prompt"]["object_name"] == "mouse"
    assert metadata["time_alignment"]["prompt_frame_idx"] == 0
    assert metadata["stage1_evidence"][0]["session_t"] == 150


def test_build_sam3_hand_clip_plan_removes_path_separators_from_prompt_text(tmp_path: Path):
    prompt = build_stage1_prompt_text(
        object_name="checkout screen",
        object_shape="rectangular",
        object_color="black",
        object_material="plastic/glass",
    )

    assert prompt == "black rectangular plastic glass checkout screen"
    assert "/" not in prompt


def test_build_sam3_hand_clip_plan_falls_back_to_hamer_raw_frames(tmp_path: Path):
    hand_window = tmp_path / "handwin_0000_t000002_000065_place_scan_operate_reach"
    frames_dir = hand_window / "hand/hamer_raw/frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_000000.jpg").write_bytes(b"fake")
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(json.dumps({"frames": []}), encoding="utf-8")
    window = {
        "window_id": "handwin_0000",
        "source_segment_range": {"start": 2, "end": 65},
        "segment_frame_range": {"start": 0, "end": 1919, "num_frames": 1920},
    }

    plan = build_sam3_hand_clip_plan(
        hand_window_dir=hand_window,
        stage1_actions=stage1,
        window=window,
        object_name="bottle",
        output_dir=tmp_path / "sam3_out",
        sam3_script=Path("/sam3/run_sam3_video.py"),
        python_executable="python",
    )

    assert plan.frames_dir == frames_dir
    assert plan.command[plan.command.index("--video") + 1] == str(frames_dir)


def test_sam3_helpers_support_hamer_raw_under_materialized_hand_window(tmp_path: Path):
    hamer_raw = tmp_path / "hand_windows/handwin_0000_t000000_000065/hand/hamer_raw"
    hamer_raw.mkdir(parents=True)

    assert infer_window_id_from_dir(hamer_raw) == "handwin_0000"
    assert default_output_dir(hamer_raw, "object_0") == (
        tmp_path / "hand_windows/handwin_0000_t000000_000065/objects/object_0"
    )


def test_build_sam3_hand_clip_plans_from_stage1_defaults_to_objects_root(tmp_path: Path):
    hand_window = tmp_path / "handwin_0000_t000002_000065"
    (hand_window / "hand/hamer_raw/frames").mkdir(parents=True)
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 4,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green", "shape": "bottle"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "window_id": "handwin_0000",
        "source_segment_range": {"start": 2, "end": 65},
        "start": 2,
        "end": 65,
        "segment_frame_range": {"start": 0, "end": 1889, "num_frames": 1890},
    }

    plans = build_sam3_hand_clip_plans_from_stage1(
        hand_window_dir=hand_window,
        stage1_actions=stage1,
        window=window,
        sam3_script=Path("/sam3/run_sam3_video.py"),
        python_executable="python",
        fps=30.0,
        clip_by_intervals=True,
    )

    assert plans[0].output_dir == (
        hand_window / "objects/bottle_green/clip_0000_t000004_000004"
    )


def test_infer_object_specs_from_stage1_uses_attributes_and_merges_duplicates(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 11,
                        "hand_object_interaction": True,
                        "interaction_objects": ["product", "phone"],
                        "interaction_object_attributes": [
                            {
                                "name": "product",
                                "color": "green and yellow",
                                "shape": "cylindrical bottle",
                                "material": "plastic",
                                "confidence": 0.95,
                            },
                            {
                                "name": "phone",
                                "color": "black",
                                "shape": "rectangular",
                                "material": "plastic and metal",
                                "confidence": 0.9,
                            },
                        ],
                    },
                    {
                        "t": 12,
                        "hand_object_interaction": True,
                        "interaction_objects": ["product"],
                        "interaction_object_attributes": [
                            {
                                "name": "product",
                                "color": "green and yellow",
                                "shape": "cylindrical bottle",
                                "material": "plastic",
                                "confidence": 0.95,
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "source_segment_range": {"start": 5, "end": 39},
        "segment_frame_range": {"start": 120, "end": 329, "num_frames": 210},
    }

    specs = infer_object_specs_from_stage1(stage1, window, fps=30.0)

    assert [spec.object_id for spec in specs] == [
        "bottle_green_and_yellow_cylindrical",
        "phone_black_rectangular",
    ]
    assert specs[0].prompt_text == "green and yellow cylindrical plastic bottle"
    assert specs[0].evidence_count == 2
    assert specs[0].active_intervals == [{"start_s": 11.0, "end_s": 12.0, "start_frame": 60, "end_frame": 90}]
    assert specs[0].prompt_frame_idx == 75
    assert specs[1].prompt_text == "black rectangular plastic and metal phone"
    assert specs[1].prompt_frame_idx == 60


def test_infer_object_specs_merges_near_duplicate_bottle_names_and_keeps_phone_interval(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 8,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "phone", "color": "black", "shape": "rectangular"},
                        ],
                    },
                    {
                        "t": 11,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green and yellow", "shape": "cylindrical bottle"},
                        ],
                    },
                    {
                        "t": 12,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green", "shape": "cylindrical bottle"},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "source_segment_range": {"start": 6, "end": 15},
        "segment_frame_range": {"start": 0, "end": 239, "num_frames": 240},
    }

    specs = infer_object_specs_from_stage1(stage1, window, fps=30.0)

    assert [spec.object_id for spec in specs] == ["bottle_green_and_yellow_cylindrical", "phone_black_rectangular"]
    assert specs[0].evidence_count == 2
    assert specs[0].prompt_frame_idx == 165
    assert specs[1].active_intervals == [{"start_s": 8.0, "end_s": 8.0, "start_frame": 60, "end_frame": 60}]
    assert specs[1].prompt_frame_idx == 60


def test_infer_object_specs_keeps_visually_different_bottles_separate(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 6,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {
                                "name": "bottle",
                                "color": "white",
                                "shape": "jug",
                                "material": "plastic",
                            },
                        ],
                    },
                    {
                        "t": 30,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {
                                "name": "product",
                                "color": "yellow and orange",
                                "shape": "rectangular package",
                                "material": "plastic and paper",
                            },
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "source_segment_range": {"start": 2, "end": 65},
        "start": 2,
        "end": 65,
        "segment_frame_range": {"start": 0, "end": 1889, "num_frames": 1890},
    }

    specs = infer_object_specs_from_stage1(stage1, window, fps=30.0)

    assert [spec.object_id for spec in specs] == [
        "bottle_white_jug",
        "product_yellow_and_orange_rectangular_package",
    ]
    assert specs[0].prompt_text == "white jug plastic bottle"
    assert specs[0].active_intervals == [{"start_s": 6.0, "end_s": 6.0, "start_frame": 120, "end_frame": 120}]
    assert specs[1].prompt_text == "yellow and orange rectangular package plastic paper product"
    assert specs[1].active_intervals == [{"start_s": 30.0, "end_s": 30.0, "start_frame": 840, "end_frame": 840}]


def test_infer_object_specs_only_uses_hand_object_interaction_true_frames(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 8,
                        "hand_object_interaction": False,
                        "interaction_object_attributes": [
                            {"name": "phone", "color": "black", "shape": "rectangular"},
                        ],
                    },
                    {
                        "t": 9,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {
                                "name": "product",
                                "hand_side": "right",
                                "color": "green",
                                "shape": "cylindrical bottle",
                                "material": "plastic",
                            },
                        ],
                    },
                    {
                        "t": 10,
                        "hand_object_interaction": False,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "red", "shape": "box"},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "source_segment_range": {"start": 6, "end": 15},
        "segment_frame_range": {"start": 0, "end": 239, "num_frames": 240},
        "objects": ["fallback_should_not_run"],
    }

    specs = infer_object_specs_from_stage1(stage1, window, fps=30.0)

    assert [spec.object_id for spec in specs] == ["bottle_green_cylindrical"]
    assert specs[0].prompt_text == "green cylindrical plastic bottle"
    assert specs[0].evidence_count == 1
    assert specs[0].active_intervals == [{"start_s": 9.0, "end_s": 9.0, "start_frame": 90, "end_frame": 90}]
    assert specs[0].prompt_frame_idx == 90


def test_infer_object_specs_returns_empty_when_window_has_no_hand_object_interaction(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 9,
                        "hand_object_interaction": False,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green", "shape": "bottle"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "source_segment_range": {"start": 6, "end": 15},
        "segment_frame_range": {"start": 0, "end": 239, "num_frames": 240},
        "objects": ["fallback_should_not_run"],
    }

    assert infer_object_specs_from_stage1(stage1, window, fps=30.0) == []


def test_build_sam3_hand_clip_plans_from_stage1_creates_one_plan_per_object(tmp_path: Path):
    hand_window = tmp_path / "handwin_0000_two_hand_left_right_000120_000329"
    (hand_window / "frames").mkdir(parents=True)
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 11,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green", "shape": "bottle"},
                            {"name": "phone", "color": "black", "shape": "rectangular"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "window_id": "handwin_0000",
        "source_segment_range": {"start": 5, "end": 39},
        "segment_frame_range": {"start": 120, "end": 329, "num_frames": 210},
        "core_segment_frame_range": {"start": 180, "end": 269, "num_frames": 90},
    }

    plans = build_sam3_hand_clip_plans_from_stage1(
        hand_window_dir=hand_window,
        stage1_actions=stage1,
        window=window,
        output_root=tmp_path / "tracks",
        sam3_script=Path("/sam3/run_sam3_video.py"),
        python_executable="/env/bin/python",
        fps=30.0,
    )

    assert [plan.object_id for plan in plans] == ["bottle_green", "phone_black_rectangular"]
    assert [plan.prompt_text for plan in plans] == ["green bottle", "black rectangular phone"]
    assert plans[0].prompt_frame_idx == 60
    assert plans[0].output_dir == tmp_path / "tracks" / "bottle_green"
    assert (plans[0].output_dir / "sam3_prompt.json").exists()


def test_build_sam3_hand_clip_plans_from_stage1_can_clip_each_object_interval(tmp_path: Path):
    hand_window = tmp_path / "handwin_0000_t000002_000065"
    frames_dir = hand_window / "hand/hamer_raw/frames"
    frames_dir.mkdir(parents=True)
    for idx in range(360):
        (frames_dir / f"frame_{idx:06d}.jpg").write_bytes(f"frame-{idx}".encode())
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 2,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green", "shape": "bottle"},
                        ],
                    },
                    {
                        "t": 5,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "cup", "color": "white", "shape": "round"},
                        ],
                    },
                    {
                        "t": 10,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "bag", "color": "transparent", "shape": "crumpled"},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "window_id": "handwin_0000",
        "source_segment_range": {"start": 2, "end": 14},
        "segment_frame_range": {"start": 0, "end": 359, "num_frames": 360},
    }

    plans = build_sam3_hand_clip_plans_from_stage1(
        hand_window_dir=hand_window,
        stage1_actions=stage1,
        window=window,
        output_root=tmp_path / "tracks",
        sam3_script=Path("/sam3/run_sam3_video.py"),
        python_executable="python",
        fps=30.0,
        clip_by_intervals=True,
        interval_expand_s=2.0,
    )

    assert [plan.object_id for plan in plans] == [
        "bottle_green",
        "cup_white_round",
        "bag_transparent_crumpled",
    ]
    assert plans[0].output_dir == tmp_path / "tracks" / "bottle_green" / "clip_0000_t000002_000004"
    assert plans[0].prompt_frame_idx == 0
    assert plans[0].command[plans[0].command.index("--video") + 1] == str(frames_dir)
    assert not (plans[0].output_dir / "input_frames").exists()
    metadata = json.loads((plans[1].output_dir / "sam3_prompt.json").read_text())
    assert metadata["time_alignment"]["object_interval"] == {
        "start_s": 3.0,
        "end_s": 7.0,
        "start_frame": 30,
        "end_frame": 150,
        "source_start_s": 5.0,
        "source_end_s": 5.0,
        "source_start_frame": 90,
        "source_end_frame": 90,
        "clip_index": 0,
    }
    assert metadata["time_alignment"]["prompt_frame_idx"] == 60


def test_interval_clips_use_hand_window_start_time_when_segment_frame_start_is_nonzero(tmp_path: Path):
    hand_window = tmp_path / "handwin_0000_t000002_000065"
    frames_dir = hand_window / "frames"
    frames_dir.mkdir(parents=True)
    for idx in range(120):
        (frames_dir / f"frame_{idx:06d}.jpg").write_bytes(b"fake")
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 4,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "scanner", "color": "black", "shape": "handheld"},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "window_id": "handwin_0000",
        "source_segment_range": {"start": 0, "end": 65},
        "start": 2,
        "end": 5,
        "segment_frame_range": {"start": 60, "end": 179, "num_frames": 120},
    }

    plans = build_sam3_hand_clip_plans_from_stage1(
        hand_window_dir=hand_window,
        stage1_actions=stage1,
        window=window,
        output_root=tmp_path / "tracks",
        sam3_script=Path("/sam3/run_sam3_video.py"),
        python_executable="python",
        fps=30.0,
        clip_by_intervals=True,
        interval_expand_s=2.0,
    )

    metadata = json.loads((plans[0].output_dir / "sam3_prompt.json").read_text())
    assert metadata["time_alignment"]["object_interval"]["start_s"] == 2.0
    assert metadata["time_alignment"]["object_interval"]["end_s"] == 5.0
    assert plans[0].output_dir.name == "clip_0000_t000002_000005"


def test_interval_clips_do_not_expand_by_default(tmp_path: Path):
    hand_window = tmp_path / "handwin_0000_t000002_000065"
    frames_dir = hand_window / "frames"
    frames_dir.mkdir(parents=True)
    for idx in range(360):
        (frames_dir / f"frame_{idx:06d}.jpg").write_bytes(b"fake")
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 6,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "bottle", "color": "white", "shape": "jug"},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "window_id": "handwin_0000",
        "source_segment_range": {"start": 2, "end": 14},
        "start": 2,
        "end": 14,
        "segment_frame_range": {"start": 0, "end": 359, "num_frames": 360},
    }

    plans = build_sam3_hand_clip_plans_from_stage1(
        hand_window_dir=hand_window,
        stage1_actions=stage1,
        window=window,
        output_root=tmp_path / "tracks",
        sam3_script=Path("/sam3/run_sam3_video.py"),
        python_executable="python",
        fps=30.0,
        clip_by_intervals=True,
    )

    metadata = json.loads((plans[0].output_dir / "sam3_prompt.json").read_text())
    assert metadata["time_alignment"]["object_interval"] == {
        "start_s": 6.0,
        "end_s": 6.0,
        "start_frame": 120,
        "end_frame": 120,
        "source_start_s": 6.0,
        "source_end_s": 6.0,
        "source_start_frame": 120,
        "source_end_frame": 120,
        "clip_index": 0,
    }
    assert plans[0].output_dir.name == "clip_0000_t000006_000006"


def test_interval_clips_merge_overlaps_and_use_each_interval_prompt_frame(tmp_path: Path):
    hand_window = tmp_path / "handwin_0000_t000002_000065"
    frames_dir = hand_window / "frames"
    frames_dir.mkdir(parents=True)
    for idx in range(600):
        (frames_dir / f"frame_{idx:06d}.jpg").write_bytes(b"fake")
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 6,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green", "shape": "bottle"},
                        ],
                    },
                    {
                        "t": 8,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green", "shape": "bottle"},
                        ],
                    },
                    {
                        "t": 10,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green", "shape": "bottle"},
                        ],
                    },
                    {
                        "t": 18,
                        "hand_object_interaction": True,
                        "interaction_object_attributes": [
                            {"name": "product", "color": "green", "shape": "bottle"},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    window = {
        "window_id": "handwin_0000",
        "source_segment_range": {"start": 2, "end": 22},
        "start": 2,
        "end": 22,
        "segment_frame_range": {"start": 0, "end": 599, "num_frames": 600},
    }

    plans = build_sam3_hand_clip_plans_from_stage1(
        hand_window_dir=hand_window,
        stage1_actions=stage1,
        window=window,
        output_root=tmp_path / "tracks",
        sam3_script=Path("/sam3/run_sam3_video.py"),
        python_executable="python",
        fps=30.0,
        clip_by_intervals=True,
        interval_expand_s=2.0,
    )

    assert [plan.output_dir.name for plan in plans] == [
        "clip_0000_t000004_000012",
        "clip_0001_t000016_000020",
    ]
    assert [plan.prompt_frame_idx for plan in plans] == [120, 60]


def test_materialize_interval_frames_writes_numbered_links_to_external_temp_dir(tmp_path: Path):
    source = tmp_path / "source_frames"
    target = tmp_path / "external_tmp"
    source.mkdir()
    for idx in range(5):
        (source / f"frame_{idx:06d}.jpg").write_bytes(f"frame-{idx}".encode())

    interval_dir = materialize_interval_frames(
        source,
        target,
        {"start_frame": 1, "end_frame": 3},
    )

    assert interval_dir == target
    assert sorted(path.name for path in target.iterdir()) == ["000000.jpg", "000001.jpg", "000002.jpg"]
    assert (target / "000000.jpg").read_bytes() == b"frame-1"


def test_combine_sam3_object_tracks_writes_index_masks_and_overlay(tmp_path: Path):
    import cv2
    import numpy as np

    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "000000.jpg"), np.full((6, 8, 3), 200, dtype=np.uint8))
    root = tmp_path / "tracks"
    for name, x in [("phone", 1), ("bottle", 4)]:
        mask_dir = root / name / "masks" / "frame_000000_masks"
        mask_dir.mkdir(parents=True)
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[2:4, x : x + 2] = 255
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask)

    out = combine_sam3_object_tracks(
        object_dirs=[root / "phone", root / "bottle"],
        frames_dir=frames,
        output_dir=root / "combined",
    )

    indexed = cv2.imread(str(Path(out["indexed_masks_dir"]) / "frame_000000_object_ids.png"), cv2.IMREAD_GRAYSCALE)
    assert sorted(set(indexed.reshape(-1).tolist())) == [0, 1, 2]
    assert (Path(out["overlays_dir"]) / "frame_000000.png").exists()
    summary = json.loads(Path(out["summary_json"]).read_text())
    assert [item["object_id"] for item in summary["objects"]] == ["phone", "bottle"]
