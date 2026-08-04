import json
from pathlib import Path

import numpy as np

from scripts.object.propose_fastsam_sam_point import qwen_candidate_rank_score, write_prompt_payload
from roboego_hand_vis.objects.fastsam_prompt import SamPointPrompt


def test_write_prompt_payload_writes_sam3_args(tmp_path: Path):
    prompt = SamPointPrompt(
        frame_idx=100,
        candidate_id="mask_3",
        point_xy=[825, 660],
        point_labels=[1],
        bbox_xywh=[800, 640, 50, 40],
        mask_area_px=1200,
        score=2.5,
        source="fastsam_hand_roi:fastsam",
    )

    payload = write_prompt_payload(
        prompt,
        tmp_path / "prompt.json",
        qwen_target={"status": "ok", "object_type": "box"},
        qwen_candidates=[{"candidate_id": "mask_3", "score": 0.9}],
    )

    saved = json.loads((tmp_path / "prompt.json").read_text())
    assert saved == payload
    assert payload["sam3_points_arg"] == "825,660"
    assert payload["sam3_point_labels_arg"] == "1"
    assert payload["sam3_box_arg"] == "800,640,850,680"
    assert payload["qwen_target"]["object_type"] == "box"
    assert payload["qwen_candidates"][0]["candidate_id"] == "mask_3"


def test_qwen_candidate_rank_score_rejects_nonmatching_large_candidate():
    large_prompt = SamPointPrompt(
        frame_idx=100,
        candidate_id="large_background",
        point_xy=[1300, 200],
        point_labels=[1],
        bbox_xywh=[80, 5, 1516, 534],
        mask_area_px=207585,
        score=207.0,
        source="fastsam",
    )
    object_prompt = SamPointPrompt(
        frame_idx=100,
        candidate_id="whole_box",
        point_xy=[830, 650],
        point_labels=[1],
        bbox_xywh=[760, 580, 150, 170],
        mask_area_px=143000,
        score=144.0,
        source="fastsam",
    )

    assert (
        qwen_candidate_rank_score(
            large_prompt,
            {
                "status": "ok",
                "is_whole_object": False,
                "matches_target": False,
                "is_part_or_pattern": True,
                "score": 0.0,
            },
        )
        is None
    )
    object_score = qwen_candidate_rank_score(
        object_prompt,
        {
            "status": "ok",
            "is_whole_object": True,
            "matches_target": True,
            "is_part_or_pattern": False,
            "score": 1.0,
        },
    )
    assert object_score is not None
    assert object_score > 1000.0
