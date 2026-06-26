from roboego_hand_vis.egohand.missing_diagnostics import (
    candidate_label,
    missing_sides_from_frame,
)


def test_missing_sides_treats_recovered_hands_as_present():
    frame = {
        "selected": {"ego_left": 2, "ego_right": None},
        "candidates": [
            {"status": "ego_left_recovered"},
            {"status": "rejected", "handedness": "right"},
        ],
    }

    assert missing_sides_from_frame(frame) == ["ego_right"]


def test_missing_sides_falls_back_to_candidate_statuses():
    frame = {
        "candidates": [
            {"status": "ego_left"},
            {"status": "ego_right_recovered"},
        ],
    }

    assert missing_sides_from_frame(frame) == []


def test_candidate_label_includes_reject_reasons_and_scores():
    label = candidate_label(
        {
            "candidate_id": 3,
            "status": "rejected",
            "handedness": "right",
            "confidence": 0.91,
            "ego_score": 0.57,
            "median_depth_m": 0.71,
            "bbox_area_ratio": 0.00068,
            "reject_reasons": ["too_small", "too_high_in_image"],
        }
    )

    assert "id=3" in label
    assert "right" in label
    assert "conf=0.91" in label
    assert "score=0.57" in label
    assert "depth=0.71m" in label
    assert "area=0.0007" in label
    assert "too_small" in label
    assert "too_high_in_image" in label
