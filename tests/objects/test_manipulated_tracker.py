import numpy as np

from roboego_hand_vis.objects.manipulated_tracker import (
    ManipulatedObjectTracker,
    ManipulatedTrackerConfig,
    MaskCandidate,
    project_joints_to_pixels,
)


INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}


def candidate(candidate_id: str, bbox_xywh: list[int], *, frame_idx: int = 0) -> MaskCandidate:
    x, y, w, h = bbox_xywh
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return MaskCandidate(
        frame_idx=frame_idx,
        candidate_id=candidate_id,
        bbox_xywh=bbox_xywh,
        mask_area_px=int((mask > 0).sum()),
        mask=mask,
        source="test_mask",
    )


def hand_at_pixel(x: float, y: float, *, z: float = 1.0) -> np.ndarray:
    return np.array(
        [[(x - INTRINSICS["cx"]) * z / INTRINSICS["fx"], (y - INTRINSICS["cy"]) * z / INTRINSICS["fy"], z]],
        dtype=np.float32,
    )


def test_project_joints_to_pixels_ignores_invalid_depth():
    joints = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 0.0], [np.nan, 0.0, 1.0]], dtype=np.float32)

    pixels = project_joints_to_pixels(joints, INTRINSICS)

    np.testing.assert_allclose(pixels, np.array([[50.0, 40.0]], dtype=np.float32))


def test_tracker_selects_candidate_near_hand_not_largest_far_candidate():
    tracker = ManipulatedObjectTracker(
        object_id="manipulated_object",
        config=ManipulatedTrackerConfig(max_hand_bbox_distance_px=12.0),
    )

    obs, selected = tracker.update(
        frame_idx=0,
        candidates=[
            candidate("far_large", [80, 70, 30, 20]),
            candidate("near_small", [42, 36, 12, 10]),
        ],
        left_joints=hand_at_pixel(50, 42),
        right_joints=None,
        intrinsics=INTRINSICS,
    )

    assert selected is not None
    assert selected.candidate_id == "near_small"
    assert obs.bbox_xywh == [42, 36, 12, 10]
    assert obs.source == "tracked:test_mask:near_small"


def test_tracker_keeps_last_bbox_through_short_mask_gap():
    tracker = ManipulatedObjectTracker(
        object_id="manipulated_object",
        config=ManipulatedTrackerConfig(max_missing_frames=2, max_hand_bbox_distance_px=12.0),
    )
    first, _ = tracker.update(
        frame_idx=0,
        candidates=[candidate("cup", [42, 36, 12, 10])],
        left_joints=hand_at_pixel(50, 42),
        right_joints=None,
        intrinsics=INTRINSICS,
    )

    missing, selected = tracker.update(
        frame_idx=1,
        candidates=[],
        left_joints=hand_at_pixel(51, 42),
        right_joints=None,
        intrinsics=INTRINSICS,
    )

    assert first.bbox_xywh == [42, 36, 12, 10]
    assert selected is None
    assert missing.bbox_xywh == [42, 36, 12, 10]
    assert missing.source == "tracked_gap_fill"


def test_tracker_switches_to_new_near_candidate_when_old_candidate_moves_away():
    tracker = ManipulatedObjectTracker(
        object_id="manipulated_object",
        config=ManipulatedTrackerConfig(max_hand_bbox_distance_px=12.0),
    )
    tracker.update(
        frame_idx=0,
        candidates=[candidate("old", [42, 36, 12, 10])],
        left_joints=hand_at_pixel(50, 42),
        right_joints=None,
        intrinsics=INTRINSICS,
    )

    obs, selected = tracker.update(
        frame_idx=1,
        candidates=[
            candidate("old", [80, 70, 12, 10], frame_idx=1),
            candidate("new", [45, 37, 12, 10], frame_idx=1),
        ],
        left_joints=hand_at_pixel(52, 43),
        right_joints=None,
        intrinsics=INTRINSICS,
    )

    assert selected is not None
    assert selected.candidate_id == "new"
    assert obs.bbox_xywh == [45, 37, 12, 10]
