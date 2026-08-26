import numpy as np

from roboego_hand_vis.egohand.final_hand_trajectory_polish import polish_depth_anchored_joints


def make_hand(frame_idx: int, side: str, wrist_x: float) -> dict:
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 0] = wrist_x
    joints[:, 1] = np.linspace(0.0, -0.08, 21)
    joints[:, 2] = 0.4
    return {
        "valid": True,
        "candidate_id": frame_idx,
        "joints21_camera_opencv": joints.astype(float).tolist(),
        "depth_anchor": {"valid": True, "depth_m": 0.4},
    }


def make_payload(values: list[float | None], side: str = "ego_left") -> dict:
    frames = []
    for frame_idx, value in enumerate(values):
        hands = {"ego_left": {"valid": False, "reason": "missing_candidate"}, "ego_right": {"valid": False, "reason": "missing_candidate"}}
        if value is not None:
            hands[side] = make_hand(frame_idx, side, value)
        frames.append({"frame_idx": frame_idx, "hands": hands})
    return {
        "stage": "joints21_depth_anchored_v1",
        "coordinate_frame": "camera_opencv",
        "unit": "meter",
        "joint_names": [f"j{i}" for i in range(21)],
        "bones": [[0, 1]],
        "frames": frames,
    }


def wrist_x(result: dict, frame_idx: int, side: str = "ego_left") -> float | None:
    hand = result["frames"][frame_idx]["hands"][side]
    if not hand.get("valid"):
        return None
    return float(hand["joints21_camera_opencv"][0][0])


def test_polish_drops_large_wrist_jump_before_interpolation():
    payload = make_payload([0.0, 0.01, 1.0, 0.03])

    result = polish_depth_anchored_joints(
        payload,
        max_interp_gap=0,
        wrist_jump_m=0.12,
        max_local_wrist_deviation_m=0,
        smooth_window=1,
    )

    assert wrist_x(result, 2) is None
    assert result["report"]["ego_left"]["dropped"] == 1
    assert result["report"]["ego_left"]["drop_reasons"]["large_wrist_jump"] == 1


def test_polish_interpolates_short_missing_gap():
    payload = make_payload([0.0, None, 0.2])

    result = polish_depth_anchored_joints(
        payload,
        max_interp_gap=1,
        wrist_jump_m=0,
        max_local_wrist_deviation_m=0,
        smooth_window=1,
    )

    assert np.isclose(wrist_x(result, 1), 0.1)
    assert result["frames"][1]["hands"]["ego_left"]["status"] == "interpolated"
    assert result["report"]["ego_left"]["interpolated"] == 1


def test_polish_smooths_wrist_jitter():
    payload = make_payload([0.0, 0.10, 0.0, 0.10, 0.0])

    result = polish_depth_anchored_joints(
        payload,
        max_interp_gap=0,
        wrist_jump_m=0,
        max_local_wrist_deviation_m=0,
        smooth_window=3,
    )

    raw = np.asarray([0.0, 0.10, 0.0, 0.10, 0.0], dtype=np.float32)
    smoothed = np.asarray([wrist_x(result, idx) for idx in range(5)], dtype=np.float32)
    assert float(smoothed.std()) < float(raw.std())
    assert result["report"]["ego_left"]["smoothed"] == 5


def test_polish_starts_new_segment_after_long_gap_instead_of_dropping():
    payload = make_payload([0.0, None, None, None, 1.0])

    result = polish_depth_anchored_joints(
        payload,
        max_interp_gap=1,
        wrist_jump_m=0.12,
        max_local_wrist_deviation_m=0,
        smooth_window=1,
    )

    assert wrist_x(result, 0) == 0.0
    assert wrist_x(result, 4) == 1.0
    assert result["report"]["ego_left"]["dropped"] == 0
    assert result["report"]["ego_left"]["intervals"] == [{"start": 0, "end": 0}, {"start": 4, "end": 4}]
