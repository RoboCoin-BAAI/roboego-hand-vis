import json
from pathlib import Path

from roboego_hand_vis.objects.sam3_hand_clip import (
    build_sam3_hand_clip_plan,
    build_sam3_hand_clip_plans_from_stage1,
    collect_stage1_evidence,
    infer_object_specs_from_stage1,
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


def test_infer_object_specs_from_stage1_uses_attributes_and_merges_duplicates(tmp_path: Path):
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 11,
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
        "product_green_and_yellow_cylindrical_bottle",
        "phone_black_rectangular",
    ]
    assert specs[0].prompt_text == "green and yellow cylindrical plastic bottle product"
    assert specs[0].evidence_count == 2
    assert specs[1].prompt_text == "black rectangular plastic and metal phone"


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

    assert [plan.object_id for plan in plans] == ["product_green_bottle", "phone_black_rectangular"]
    assert [plan.prompt_text for plan in plans] == ["green bottle product", "black rectangular phone"]
    assert plans[0].output_dir == tmp_path / "tracks" / "handwin_0000" / "product_green_bottle"
    assert (plans[0].output_dir / "sam3_prompt.json").exists()
