import json
from pathlib import Path

import cv2
import numpy as np

from scripts.object.propose_fastsam_sam_point import (
    build_qwen_semantic_pool,
    build_qwen_semantic_pool_rows,
    build_qwen_candidate_prompt_pool,
    compute_yellow_mask_score,
    choose_qwen_prompt_for_candidate,
    prompt_to_sam3_object,
    qwen_candidate_rank_score,
    save_candidate_debug_artifacts,
    select_prompt_from_qwen_semantic_pool,
    write_multi_object_prompt_payload,
    write_qwen_ranked_candidates,
    write_prompt_payload,
)
from roboego_hand_vis.objects.fastsam_prompt import SamPointPrompt
from roboego_hand_vis.objects.manipulated_tracker import MaskCandidate


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


def test_compute_yellow_mask_score_prefers_yellow_pixels_inside_mask():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = [20, 20, 20]
    image[1:5, 1:5] = [245, 210, 30]
    yellow_mask = np.zeros((8, 8), dtype=np.uint8)
    yellow_mask[1:5, 1:5] = 255
    dark_mask = np.zeros((8, 8), dtype=np.uint8)
    dark_mask[5:8, 5:8] = 255

    yellow_score = compute_yellow_mask_score(image, yellow_mask)
    dark_score = compute_yellow_mask_score(image, dark_mask)

    assert yellow_score["yellow_ratio"] > 0.9
    assert yellow_score["yellow_pixels"] == 16
    assert dark_score["yellow_ratio"] == 0.0


def test_build_qwen_candidate_prompt_pool_keeps_strong_yellow_candidate_when_hand_score_is_low():
    def prompt(candidate_id: str, score: float) -> SamPointPrompt:
        return SamPointPrompt(
            frame_idx=100,
            candidate_id=candidate_id,
            point_xy=[800, 650],
            point_labels=[1],
            bbox_xywh=[760, 580, 150, 170],
            mask_area_px=1000,
            score=score,
            source="fastsam",
        )

    pool = build_qwen_candidate_prompt_pool(
        [
            (prompt("near_not_yellow", 20.0), {"yellow_ratio": 0.01, "yellow_pixels": 10}),
            (prompt("mid_not_yellow", 10.0), {"yellow_ratio": 0.02, "yellow_pixels": 20}),
            (prompt("yellow_box", -80.0), {"yellow_ratio": 0.84, "yellow_pixels": 18063}),
        ],
        top_k=2,
    )

    assert [item[0].candidate_id for item in pool] == ["near_not_yellow", "yellow_box"]


def test_choose_qwen_prompt_for_candidate_allows_yellow_candidate_farther_from_hand():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:70, 50:80] = 255
    candidate = MaskCandidate(
        frame_idx=100,
        candidate_id="yellow_box",
        bbox_xywh=[50, 50, 30, 20],
        mask_area_px=600,
        mask=mask,
        source="fastsam",
    )
    joints = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32).reshape(1, 3)

    prompt = choose_qwen_prompt_for_candidate(
        candidate,
        left_joints=joints,
        right_joints=None,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )

    assert prompt is not None
    assert prompt.candidate_id == "yellow_box"


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


def test_select_prompt_from_qwen_semantic_pool_trusts_semantic_ranking_before_hand_geometry():
    def prompt(candidate_id: str, score: float) -> SamPointPrompt:
        return SamPointPrompt(
            frame_idx=100,
            candidate_id=candidate_id,
            point_xy=[800, 650],
            point_labels=[1],
            bbox_xywh=[760, 580, 150, 170],
            mask_area_px=1000,
            score=score,
            source="fastsam",
        )

    selected = select_prompt_from_qwen_semantic_pool(
        [
            (prompt("semantic_best_far_hand", -80.0), {"status": "ok", "score": 0.99}),
            (prompt("semantic_second_near_hand", -2.0), {"status": "ok", "score": 0.95}),
            (prompt("semantic_third_mid_hand", -12.0), {"status": "ok", "score": 0.90}),
            (prompt("semantic_fourth_nearest_hand", 100.0), {"status": "ok", "score": 0.50}),
        ],
        semantic_pool_size=3,
    )

    assert selected is not None
    selected_prompt, selected_qwen = selected
    assert selected_prompt.candidate_id == "semantic_best_far_hand"
    assert selected_qwen["score"] == 0.99


def test_build_qwen_semantic_pool_breaks_qwen_score_ties_with_yellow_score():
    def prompt(candidate_id: str) -> SamPointPrompt:
        return SamPointPrompt(
            frame_idx=100,
            candidate_id=candidate_id,
            point_xy=[800, 650],
            point_labels=[1],
            bbox_xywh=[760, 580, 150, 170],
            mask_area_px=1000,
            score=0.0,
            source="fastsam",
        )

    semantic_pool = build_qwen_semantic_pool(
        [
            (prompt("low_yellow"), {"status": "ok", "score": 1.0, "yellow_ratio": 0.1, "yellow_pixels": 100}),
            (prompt("high_yellow"), {"status": "ok", "score": 1.0, "yellow_ratio": 0.8, "yellow_pixels": 1000}),
        ],
        semantic_pool_size=1,
    )

    assert semantic_pool[0][0].candidate_id == "high_yellow"


def test_build_qwen_semantic_pool_rows_preserves_hand_geometry_score():
    selected = SamPointPrompt(
        frame_idx=100,
        candidate_id="selected",
        point_xy=[800, 650],
        point_labels=[1],
        bbox_xywh=[760, 580, 150, 170],
        mask_area_px=1000,
        score=-2.0,
        source="fastsam",
    )

    rows = build_qwen_semantic_pool_rows(
        [(selected, {"status": "ok", "score": 0.95})],
        selected_candidate_id="selected",
    )

    selected.score = 950.0
    assert rows[0]["hand_geometry_score"] == -2.0
    assert rows[0]["selected_by_semantic_rank"] is True
    assert rows[0]["selected_by_hand_geometry"] is True


def test_write_qwen_ranked_candidates_writes_sidecar_json(tmp_path: Path):
    rows = [
        {
            "candidate_id": "fastsam_1",
            "crop_path": "qwen_candidate_crops/fastsam_1.png",
            "score": 0.98,
            "reason": "whole yellow box",
            "hand_geometry_score": -2.0,
            "selected_by_hand_geometry": True,
        },
        {
            "candidate_id": "fastsam_64",
            "crop_path": "qwen_candidate_crops/fastsam_64.png",
            "score": 0.95,
            "reason": "contains lower desk area",
            "hand_geometry_score": -20.0,
            "selected_by_hand_geometry": False,
        },
    ]

    output_path = write_qwen_ranked_candidates(rows, tmp_path)

    saved = json.loads(output_path.read_text())
    assert output_path.name == "qwen_ranked_candidates.json"
    assert saved[0]["candidate_id"] == "fastsam_1"
    assert saved[0]["crop_path"].endswith("fastsam_1.png")
    assert saved[0]["selected_by_hand_geometry"] is True


def test_write_multi_object_prompt_payload_writes_top3_sam3_objects(tmp_path: Path):
    prompts = [
        SamPointPrompt(
            frame_idx=100,
            candidate_id=f"fastsam_{idx}",
            point_xy=[800 + idx, 650],
            point_labels=[1],
            bbox_xywh=[760, 580, 150, 170],
            mask_area_px=1000,
            score=0.9 - idx * 0.1,
            source="fastsam:qwen",
        )
        for idx in range(3)
    ]
    rows = [
        {
            "candidate_id": f"fastsam_{idx}",
            "score": 0.99 - idx * 0.01,
            "yellow_ratio": 0.8 - idx * 0.1,
            "reason": "whole yellow box",
        }
        for idx in range(3)
    ]

    payload = write_multi_object_prompt_payload(
        [prompt_to_sam3_object(prompt, rows[idx], object_index=idx) for idx, prompt in enumerate(prompts)],
        tmp_path / "multi_object_prompts.json",
    )

    saved = json.loads((tmp_path / "multi_object_prompts.json").read_text())
    assert saved == payload
    assert [obj["obj_id"] for obj in payload["objects"]] == [
        "manipulated_object_0",
        "manipulated_object_1",
        "manipulated_object_2",
    ]
    assert payload["objects"][0]["sam3_points_arg"] == "800,650"
    assert payload["objects"][0]["yellow_ratio"] == 0.8


def test_save_candidate_debug_artifacts_writes_crop_mask_overlay_and_index(tmp_path: Path):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :] = [10, 20, 30]
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:10, 7:13] = 255
    candidate = type(
        "Candidate",
        (),
        {
            "candidate_id": "fastsam_0",
            "frame_idx": 4,
            "bbox_xywh": [7, 5, 6, 5],
            "mask_area_px": 30,
            "mask": mask,
            "source": "fastsam",
        },
    )()

    rows = save_candidate_debug_artifacts(
        image_rgb=image,
        candidates=[candidate],
        output_dir=tmp_path / "candidates",
    )

    assert rows[0]["candidate_id"] == "fastsam_0"
    assert rows[0]["crop_path"] == "fastsam_0_crop.png"
    assert (tmp_path / "candidates" / "fastsam_0_crop.png").exists()
    assert (tmp_path / "candidates" / "fastsam_0_mask.png").exists()
    assert (tmp_path / "candidates" / "fastsam_0_overlay.png").exists()
    assert json.loads((tmp_path / "candidates" / "candidates.json").read_text())[0]["mask_area_px"] == 30
    loaded = cv2.imread(str(tmp_path / "candidates" / "fastsam_0_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert int((loaded > 0).sum()) == 30
