import numpy as np

from roboego_hand_vis.egohand.joints21_pose_gate import build_joints21_pose_quality


def make_joints(offset_x: float = 0.0) -> np.ndarray:
    joints = np.zeros((21, 3), dtype=np.float32)
    for finger, base in enumerate([1, 5, 9, 13, 17]):
        x = offset_x + finger * 0.02
        for step in range(4):
            joints[base + step] = [x, -0.03 * (step + 1), 0.0]
    return joints


def make_hand(candidate_id: int, side: str, joints: np.ndarray) -> dict:
    return {
        "frame_idx": 0,
        "side": side,
        "candidate_id": candidate_id,
        "status": "normal",
        "bbox_xyxy": [10.0, 20.0, 60.0, 100.0],
        "confidence": 0.9,
        "mesh_obj": f"/tmp/frame_000000_{side}.obj",
        "joints21_camera_opencv": joints.astype(float).tolist(),
        "source_hand": {
            "candidate_id": candidate_id,
            "mano_params": {"large": "not needed downstream"},
            "mesh_obj": f"/tmp/original_frame_000000_{side}.obj",
        },
    }


def make_payload(hands_by_frame: list[dict[str, dict | None]]) -> dict:
    frames = []
    for frame_idx, hands in enumerate(hands_by_frame):
        frame_hands = {}
        for side in ("ego_left", "ego_right"):
            hand = hands.get(side)
            if hand is not None:
                hand = dict(hand)
                hand["frame_idx"] = frame_idx
            frame_hands[side] = hand
        frames.append({"frame_idx": frame_idx, "hands": frame_hands})
    return {
        "stage": "final_hand_result_v1",
        "frames": frames,
        "tracks": {
            "ego_left": [frame["hands"]["ego_left"] for frame in frames],
            "ego_right": [frame["hands"]["ego_right"] for frame in frames],
        },
    }


def test_pose_gate_exports_kept_hands_as_alignment_candidates():
    joints = make_joints()
    payload = make_payload([{"ego_left": make_hand(1, "ego_left", joints)}])

    result = build_joints21_pose_quality(
        payload,
        start=0,
        num_frames=1,
        image_height=120,
        max_suspicious_wrist_y_ratio=0.55,
    )

    assert result["summary"]["ego_left"]["kept"] == 1
    assert result["summary"]["ego_left"]["dropped"] == 0
    candidate = result["final_hand_candidates_gated"]["frames"][0]["candidates"][0]
    assert candidate["status"] == "ego_left"
    assert candidate["candidate_id"] == 1
    assert candidate["mesh_obj"] == "/tmp/frame_000000_ego_left.obj"
    assert "source_hand" not in candidate
    assert "mano_params" not in candidate


def test_pose_gate_drops_bad_bone_length_ratio():
    joints = make_joints()
    joints[8] = [0.1, 10.0, 0.0]
    payload = make_payload([{"ego_left": make_hand(1, "ego_left", joints)}])

    result = build_joints21_pose_quality(payload, start=0, num_frames=1, max_bone_length_ratio=8.0)

    assert result["summary"]["ego_left"]["kept"] == 0
    assert result["summary"]["ego_left"]["dropped"] == 1
    dropped = result["frames"][0]["hands"]["ego_left"]
    assert dropped["valid"] is False
    assert dropped["drop_reasons"] == ["bone_length_outlier"]
    assert result["final_hand_candidates_gated"]["frames"][0]["candidates"] == []


def test_pose_gate_drops_local_joint_shape_outlier_without_cascading():
    good0 = make_joints(offset_x=0.0)
    bad = make_joints(offset_x=0.01)
    bad[4:] += np.array([1.0, 0.0, 0.0], dtype=np.float32)
    good2 = make_joints(offset_x=0.02)
    payload = make_payload(
        [
            {"ego_left": make_hand(1, "ego_left", good0)},
            {"ego_left": make_hand(2, "ego_left", bad)},
            {"ego_left": make_hand(3, "ego_left", good2)},
        ]
    )

    result = build_joints21_pose_quality(
        payload,
        start=0,
        num_frames=3,
        local_window_radius=1,
        max_joint_local_deviation=5.0,
    )

    assert result["frames"][0]["hands"]["ego_left"]["valid"] is True
    assert "large_joint_local_deviation" in result["frames"][1]["hands"]["ego_left"]["drop_reasons"]
    assert result["frames"][2]["hands"]["ego_left"]["valid"] is True
    assert [len(frame["candidates"]) for frame in result["final_hand_candidates_gated"]["frames"]] == [1, 0, 1]


def test_pose_gate_drops_finger_direction_flip():
    good0 = make_joints(offset_x=0.0)
    bad = make_joints(offset_x=0.01)
    bad[5:9, 1] *= -1.0
    good2 = make_joints(offset_x=0.02)
    payload = make_payload(
        [
            {"ego_left": make_hand(1, "ego_left", good0)},
            {"ego_left": make_hand(2, "ego_left", bad)},
            {"ego_left": make_hand(3, "ego_left", good2)},
        ]
    )

    result = build_joints21_pose_quality(
        payload,
        start=0,
        num_frames=3,
        local_window_radius=1,
        min_finger_direction_dot=0.0,
    )

    assert result["frames"][1]["hands"]["ego_left"]["drop_reasons"] == ["finger_direction_flip"]


def test_pose_gate_drops_hand_when_fingertips_are_below_wrist_in_image():
    joints = make_joints()
    hand = make_hand(1, "ego_left", joints)
    hand["joints21_image_xy"] = [[50.0, 40.0] for _ in range(21)]
    hand["joints21_image_xy"][0] = [50.0, 40.0]
    for idx in [4, 8, 12, 16, 20]:
        hand["joints21_image_xy"][idx] = [50.0, 100.0]
    payload = make_payload([{"ego_left": hand}])

    result = build_joints21_pose_quality(
        payload,
        start=0,
        num_frames=1,
        image_height=120,
        max_suspicious_wrist_y_ratio=0.55,
    )

    assert result["summary"]["ego_left"]["kept"] == 0
    assert result["summary"]["ego_left"]["dropped"] == 1
    dropped = result["frames"][0]["hands"]["ego_left"]
    assert dropped["drop_reasons"] == ["fingertips_below_wrist"]
    assert result["final_hand_candidates_gated"]["frames"][0]["candidates"] == []


def test_pose_gate_keeps_hand_when_only_four_fingertips_are_below_wrist():
    joints = make_joints()
    hand = make_hand(1, "ego_left", joints)
    hand["joints21_image_xy"] = [[50.0, 40.0] for _ in range(21)]
    hand["joints21_image_xy"][0] = [50.0, 40.0]
    for idx in [4, 8, 12, 16]:
        hand["joints21_image_xy"][idx] = [50.0, 100.0]
    hand["joints21_image_xy"][20] = [50.0, 20.0]
    payload = make_payload([{"ego_left": hand}])

    result = build_joints21_pose_quality(
        payload,
        start=0,
        num_frames=1,
        image_height=120,
        max_suspicious_wrist_y_ratio=0.55,
    )

    assert result["summary"]["ego_left"]["kept"] == 1
    assert result["frames"][0]["hands"]["ego_left"]["valid"] is True


def test_pose_gate_keeps_lower_frame_hand_even_when_fingertips_are_below_wrist():
    joints = make_joints()
    hand = make_hand(1, "ego_left", joints)
    hand["joints21_image_xy"] = [[50.0, 80.0] for _ in range(21)]
    hand["joints21_image_xy"][0] = [50.0, 80.0]
    for idx in [4, 8, 12, 16, 20]:
        hand["joints21_image_xy"][idx] = [50.0, 100.0]
    payload = make_payload([{"ego_left": hand}])

    result = build_joints21_pose_quality(
        payload,
        start=0,
        num_frames=1,
        image_height=120,
        max_suspicious_wrist_y_ratio=0.55,
    )

    assert result["summary"]["ego_left"]["kept"] == 1
    assert result["frames"][0]["hands"]["ego_left"]["valid"] is True


def test_pose_gate_drops_fingertip_trajectory_jump_without_cascading():
    good0 = make_joints(offset_x=0.0)
    bad = make_joints(offset_x=0.01)
    bad[[4, 8, 12, 16, 20]] += np.array([0.8, 0.0, 0.0], dtype=np.float32)
    good2 = make_joints(offset_x=0.02)
    payload = make_payload(
        [
            {"ego_left": make_hand(1, "ego_left", good0)},
            {"ego_left": make_hand(2, "ego_left", bad)},
            {"ego_left": make_hand(3, "ego_left", good2)},
        ]
    )

    result = build_joints21_pose_quality(
        payload,
        start=0,
        num_frames=3,
        local_window_radius=1,
        max_fingertip_local_deviation_m=0.2,
    )

    assert result["frames"][0]["hands"]["ego_left"]["valid"] is True
    assert "large_fingertip_trajectory_deviation" in result["frames"][1]["hands"]["ego_left"]["drop_reasons"]
    assert result["frames"][2]["hands"]["ego_left"]["valid"] is True


def test_pose_gate_drops_wrist_and_palm_trajectory_jump():
    good0 = make_joints(offset_x=0.0)
    bad = make_joints(offset_x=0.01)
    bad[[0, 5, 9, 13, 17]] += np.array([0.0, 0.5, 0.0], dtype=np.float32)
    good2 = make_joints(offset_x=0.02)
    payload = make_payload(
        [
            {"ego_left": make_hand(1, "ego_left", good0)},
            {"ego_left": make_hand(2, "ego_left", bad)},
            {"ego_left": make_hand(3, "ego_left", good2)},
        ]
    )

    result = build_joints21_pose_quality(
        payload,
        start=0,
        num_frames=3,
        local_window_radius=1,
        max_wrist_local_deviation_m=0.2,
        max_palm_local_deviation_m=0.2,
    )

    reasons = result["frames"][1]["hands"]["ego_left"]["drop_reasons"]
    assert "large_wrist_trajectory_deviation" in reasons
    assert "large_palm_trajectory_deviation" in reasons
