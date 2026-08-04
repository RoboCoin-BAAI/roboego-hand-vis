import numpy as np

from roboego_hand_vis.objects.fastsam_prompt import (
    FastSamPromptConfig,
    choose_sam_point_from_candidates,
)
from roboego_hand_vis.objects.manipulated_tracker import MaskCandidate


INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}


def candidate(candidate_id: str, bbox_xywh: list[int]) -> MaskCandidate:
    x, y, w, h = bbox_xywh
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return MaskCandidate(
        frame_idx=0,
        candidate_id=candidate_id,
        bbox_xywh=bbox_xywh,
        mask_area_px=int((mask > 0).sum()),
        mask=mask,
        source="fastsam",
    )


def hand_at_pixel(x: float, y: float, *, z: float = 1.0) -> np.ndarray:
    return np.array(
        [[(x - INTRINSICS["cx"]) * z / INTRINSICS["fx"], (y - INTRINSICS["cy"]) * z / INTRINSICS["fy"], z]],
        dtype=np.float32,
    )


def test_choose_sam_point_prefers_object_near_hand_over_largest_far_mask():
    prompt = choose_sam_point_from_candidates(
        candidates=[
            candidate("far_large", [80, 70, 32, 22]),
            candidate("near_object", [48, 44, 16, 14]),
        ],
        left_joints=hand_at_pixel(50, 42),
        right_joints=None,
        intrinsics=INTRINSICS,
        config=FastSamPromptConfig(max_hand_bbox_distance_px=16.0),
    )

    assert prompt is not None
    assert prompt.candidate_id == "near_object"
    assert prompt.point_xy == [55, 50]


def test_choose_sam_point_rejects_mask_containing_too_many_hand_joints():
    joints = np.concatenate(
        [
            hand_at_pixel(50, 42),
            hand_at_pixel(51, 42),
            hand_at_pixel(52, 42),
        ],
        axis=0,
    )

    prompt = choose_sam_point_from_candidates(
        candidates=[
            candidate("hand_mask", [45, 37, 14, 12]),
            candidate("near_object", [60, 40, 14, 12]),
        ],
        left_joints=joints,
        right_joints=None,
        intrinsics=INTRINSICS,
        config=FastSamPromptConfig(max_hand_bbox_distance_px=18.0, max_hand_points_inside_ratio=0.4),
    )

    assert prompt is not None
    assert prompt.candidate_id == "near_object"


def test_choose_sam_point_returns_none_without_near_candidate():
    prompt = choose_sam_point_from_candidates(
        candidates=[candidate("far", [90, 70, 20, 20])],
        left_joints=hand_at_pixel(50, 42),
        right_joints=None,
        intrinsics=INTRINSICS,
        config=FastSamPromptConfig(max_hand_bbox_distance_px=10.0),
    )

    assert prompt is None


def test_choose_sam_point_rejects_oversized_background_mask():
    prompt = choose_sam_point_from_candidates(
        candidates=[
            candidate("huge_background", [0, 0, 118, 98]),
            candidate("near_object", [48, 44, 16, 14]),
        ],
        left_joints=hand_at_pixel(50, 42),
        right_joints=None,
        intrinsics=INTRINSICS,
        config=FastSamPromptConfig(
            max_hand_bbox_distance_px=20.0,
            max_mask_frame_area_ratio=0.45,
            area_weight=0.001,
        ),
    )

    assert prompt is not None
    assert prompt.candidate_id == "near_object"
