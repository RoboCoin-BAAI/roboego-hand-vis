import numpy as np
import pytest

from roboego_hand_vis.egohand.filtering import (
    EgoHandFilterConfig,
    bbox_median_depth,
    select_ego_hands,
)


def test_selects_one_ego_hand_per_side_and_keeps_extra_valid_hands_as_other():
    depth = np.full((100, 100), 1.0, dtype=np.float32)
    depth[70:95, 5:35] = 0.45
    depth[70:95, 65:95] = 0.55
    depth[65:85, 42:62] = 0.75

    candidates = [
        {"bbox_xyxy": [5, 70, 35, 95], "handedness": "left", "confidence": 0.9},
        {"bbox_xyxy": [65, 70, 95, 95], "handedness": "right", "confidence": 0.85},
        {"bbox_xyxy": [42, 65, 62, 85], "handedness": "right", "confidence": 0.7},
    ]

    result = select_ego_hands(
        candidates,
        image_width=100,
        image_height=100,
        depth_m=depth,
        config=EgoHandFilterConfig(min_area_ratio=0.02, preferred_area_ratio=0.08),
    )

    assert result.selected["ego_left"] == 0
    assert result.selected["ego_right"] == 1
    assert result.candidates[2]["status"] == "other_hands"
    assert result.candidates[0]["median_depth_m"] == pytest.approx(0.45)
    assert result.candidates[1]["median_depth_m"] == pytest.approx(0.55)


def test_rejects_low_confidence_tiny_top_or_far_candidates():
    depth = np.full((100, 100), 1.0, dtype=np.float32)
    depth[80:90, 80:90] = 3.5

    candidates = [
        {"bbox_xyxy": [10, 80, 40, 96], "handedness": "left", "confidence": 0.2},
        {"bbox_xyxy": [10, 80, 12, 82], "handedness": "right", "confidence": 0.9},
        {"bbox_xyxy": [10, 2, 40, 20], "handedness": "left", "confidence": 0.9},
        {"bbox_xyxy": [80, 80, 90, 90], "handedness": "right", "confidence": 0.9},
    ]

    result = select_ego_hands(
        candidates,
        image_width=100,
        image_height=100,
        depth_m=depth,
        config=EgoHandFilterConfig(
            min_area_ratio=0.01,
            min_center_y_ratio=0.25,
            min_confidence=0.5,
            max_median_depth_m=3.0,
        ),
    )

    assert result.selected == {"ego_left": None, "ego_right": None}
    assert [candidate["status"] for candidate in result.candidates] == [
        "rejected",
        "rejected",
        "rejected",
        "rejected",
    ]
    assert "low_confidence" in result.candidates[0]["reject_reasons"]
    assert "too_small" in result.candidates[1]["reject_reasons"]
    assert "too_high_in_image" in result.candidates[2]["reject_reasons"]
    assert "too_far" in result.candidates[3]["reject_reasons"]


def test_bbox_median_depth_scales_image_bbox_to_depth_resolution():
    depth = np.ones((50, 50), dtype=np.float32)
    depth[20:30, 20:30] = 0.4

    median = bbox_median_depth(
        depth,
        bbox_xyxy=[40, 40, 60, 60],
        image_width=100,
        image_height=100,
    )

    assert median == pytest.approx(0.4)


def test_recovers_missing_side_from_near_confident_small_candidate():
    depth = np.full((100, 100), 1.0, dtype=np.float32)
    depth[70:95, 5:35] = 0.45
    depth[50:62, 62:74] = 0.6
    depth[5:12, 20:27] = 1.4

    candidates = [
        {"bbox_xyxy": [5, 70, 35, 95], "handedness": "left", "confidence": 0.9},
        {"bbox_xyxy": [62, 50, 74, 62], "handedness": "right", "confidence": 0.92},
        {"bbox_xyxy": [20, 5, 27, 12], "handedness": "right", "confidence": 0.9},
    ]

    result = select_ego_hands(
        candidates,
        image_width=100,
        image_height=100,
        depth_m=depth,
        config=EgoHandFilterConfig(
            min_area_ratio=0.02,
            recovery_min_area_ratio=0.01,
            recovery_min_center_y_ratio=0.35,
        ),
    )

    assert result.selected["ego_left"] == 0
    assert result.selected["ego_right"] == 1
    assert result.candidates[1]["status"] == "ego_right_recovered"
    assert result.candidates[1]["recovery"]["reason"] == "missing_side_fallback"
    assert result.candidates[2]["status"] == "rejected"


def test_select_ego_hands_uses_wearer_bbox_as_hard_roi():
    candidates = [
        {"bbox_xyxy": [5, 70, 35, 95], "handedness": "left", "confidence": 0.9},
        {"bbox_xyxy": [65, 70, 95, 95], "handedness": "right", "confidence": 0.9},
        {"bbox_xyxy": [10, 10, 40, 35], "handedness": "left", "confidence": 0.99},
    ]

    result = select_ego_hands(
        candidates,
        image_width=100,
        image_height=100,
        wearer_bbox_xyxy=[0.0, 0.5, 1.0, 1.0],
        config=EgoHandFilterConfig(min_area_ratio=0.01),
    )

    assert result.selected["ego_left"] == 0
    assert result.selected["ego_right"] == 1
    assert result.candidates[2]["status"] == "rejected"
    assert "outside_wearer_region" in result.candidates[2]["reject_reasons"]


def test_wearer_bbox_gate_expands_vertically_but_not_horizontally():
    candidates = [
        {"bbox_xyxy": [40, 76, 60, 86], "handedness": "left", "confidence": 0.9},
        {"bbox_xyxy": [20, 88, 34, 98], "handedness": "right", "confidence": 0.9},
    ]

    result = select_ego_hands(
        candidates,
        image_width=100,
        image_height=100,
        wearer_bbox_xyxy=[0.35, 0.85, 0.65, 1.0],
        config=EgoHandFilterConfig(min_area_ratio=0.01, wearer_roi_y_margin_ratio=0.05),
    )

    assert result.selected["ego_left"] == 0
    assert result.selected["ego_right"] is None
    assert "outside_wearer_region" in result.candidates[1]["reject_reasons"]


def test_wearer_bbox_gate_can_expand_x_when_explicitly_configured():
    candidates = [
        {"bbox_xyxy": [20, 88, 34, 98], "handedness": "right", "confidence": 0.9},
    ]

    strict = select_ego_hands(
        candidates,
        image_width=100,
        image_height=100,
        wearer_bbox_xyxy=[0.35, 0.85, 0.65, 1.0],
        config=EgoHandFilterConfig(
            min_area_ratio=0.01,
            wearer_roi_x_margin_ratio=0.0,
            wearer_roi_y_margin_ratio=0.05,
        ),
    )
    expanded = select_ego_hands(
        candidates,
        image_width=100,
        image_height=100,
        wearer_bbox_xyxy=[0.35, 0.85, 0.65, 1.0],
        config=EgoHandFilterConfig(
            min_area_ratio=0.01,
            wearer_roi_x_margin_ratio=0.2,
            wearer_roi_y_margin_ratio=0.05,
        ),
    )

    assert strict.selected["ego_right"] is None
    assert expanded.selected["ego_right"] == 0


def test_semantic_hand_visible_enables_more_permissive_roi_and_area_recovery():
    candidates = [
        {"bbox_xyxy": [40, 58, 60, 68], "handedness": "left", "confidence": 0.9},
        {"bbox_xyxy": [18, 88, 32, 98], "handedness": "right", "confidence": 0.9},
    ]
    config = EgoHandFilterConfig(
        min_area_ratio=0.02,
        recovery_min_area_ratio=0.005,
        wearer_roi_y_margin_ratio=0.05,
        semantic_hand_visible_y_margin_ratio=0.3,
        semantic_hand_visible_min_area_ratio=0.005,
    )

    strict_result = select_ego_hands(
        candidates,
        image_width=100,
        image_height=100,
        wearer_bbox_xyxy=[0.35, 0.85, 0.65, 1.0],
        config=config,
        semantic_hand_visible=False,
    )
    semantic_result = select_ego_hands(
        candidates,
        image_width=100,
        image_height=100,
        wearer_bbox_xyxy=[0.35, 0.85, 0.65, 1.0],
        config=config,
        semantic_hand_visible=True,
    )

    assert strict_result.selected["ego_left"] is None
    assert semantic_result.selected["ego_left"] == 0
    assert semantic_result.selected["ego_right"] is None
    assert "outside_wearer_region" in semantic_result.candidates[1]["reject_reasons"]
