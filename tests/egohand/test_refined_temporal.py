from pathlib import Path

from roboego_hand_vis.egohand.refined_temporal import build_temporal_tracks_from_refined


def hand(candidate_id: int, bbox: list[float], side: str = "left") -> dict:
    return {
        "candidate_id": candidate_id,
        "bbox_xyxy": bbox,
        "handedness": side,
        "refine_quality": "reliable",
        "confidence": 0.9,
        "detector_score": 0.9,
        "keypoint_mean_conf": 0.9,
        "valid_keypoints": 21,
        "pred_cam_t_full": [0.0, 0.0, 1.0],
        "mano_params": {
            "global_orient_rotmat": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            "hand_pose_rotmat": [],
        },
    }


def hand_with_pose(candidate_id: int, bbox: list[float], pose_value: float, side: str = "left") -> dict:
    item = hand(candidate_id, bbox, side=side)
    item["mano_params"] = {
        "global_orient_rotmat": [
            [[1.0 + pose_value, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ],
        "hand_pose_rotmat": [
            [[1.0, pose_value, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
    }
    return item


def write_mesh(path: Path, vertices: list[list[float]]) -> str:
    path.write_text(
        "\n".join(f"v {x} {y} {z}" for x, y, z in vertices) + "\n",
        encoding="utf-8",
    )
    return str(path)


def test_refined_temporal_keeps_reliable_refine_observations_as_normal():
    payload = {"frames": [{"frame_idx": 10, "ego_left_refined": hand(1, [10, 20, 30, 60])}]}

    result = build_temporal_tracks_from_refined(payload, start=10, num_frames=1)

    obs = result["tracks"]["ego_left"][0]
    assert obs["status"] == "normal"
    assert obs["source"] == "strict_refine_reliable"
    assert obs["candidate_id"] == 1
    assert result["summary"]["ego_left"]["normal"] == 1


def test_refined_temporal_does_not_interpolate_short_missing_gap():
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_left_refined": hand(1, [0, 0, 10, 10])},
            {"frame_idx": 2, "ego_left_refined": hand(2, [20, 0, 30, 10])},
        ]
    }

    result = build_temporal_tracks_from_refined(payload, start=0, num_frames=3, max_gap=2)

    track = result["tracks"]["ego_left"]
    assert track[0]["status"] == "normal"
    assert track[1] is None
    assert track[2]["status"] == "normal"
    assert result["summary"]["ego_left"]["interpolated"] == 0
    assert result["summary"]["ego_left"]["missing"] == 1


def test_refined_temporal_drops_large_bbox_jump_when_area_also_changes():
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_right_refined": hand(1, [0, 0, 10, 10], side="right")},
            {"frame_idx": 1, "ego_right_refined": hand(2, [500, 0, 560, 60], side="right")},
        ]
    }

    result = build_temporal_tracks_from_refined(
        payload,
        start=0,
        num_frames=2,
        max_bbox_jump_px=100,
        max_bbox_area_change_ratio=3.0,
    )

    assert result["tracks"]["ego_right"][1] is None
    assert result["frames"][1]["dropped_hands"]["ego_right"]["drop_reasons"] == [
        "large_bbox_area_change",
        "large_bbox_center_jump",
    ]
    assert result["summary"]["ego_right"]["dropped"] == 1


def test_refined_temporal_keeps_large_bbox_jump_as_warning_when_area_is_stable():
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_right_refined": hand(1, [0, 0, 10, 10], side="right")},
            {"frame_idx": 1, "ego_right_refined": hand(2, [500, 0, 510, 10], side="right")},
        ]
    }

    result = build_temporal_tracks_from_refined(payload, start=0, num_frames=2, max_bbox_jump_px=100)

    obs = result["tracks"]["ego_right"][1]
    assert obs["status"] == "normal"
    assert obs["temporal_quality"] == "jump_warning"
    assert "large_bbox_center_jump" in obs["anomaly_flags"]
    assert result["summary"]["ego_right"]["large_bbox_center_jump"] == 1


def test_refined_temporal_drops_nonfinite_pose_values():
    bad = hand(2, [20, 0, 30, 10], side="right")
    bad["mano_params"]["global_orient_rotmat"] = [[[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_right_refined": hand(1, [0, 0, 10, 10], side="right")},
            {"frame_idx": 1, "ego_right_refined": bad},
        ]
    }

    result = build_temporal_tracks_from_refined(payload, start=0, num_frames=2)

    assert result["tracks"]["ego_right"][1] is None
    assert result["frames"][1]["dropped_hands"]["ego_right"]["drop_reasons"] == ["nonfinite_mano_pose"]


def test_refined_temporal_drops_large_camera_translation_jump():
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_right_refined": hand(1, [0, 0, 10, 10], side="right")},
            {
                "frame_idx": 1,
                "ego_right_refined": {
                    **hand(2, [1, 0, 11, 10], side="right"),
                    "pred_cam_t_full": [0.0, 0.0, 10.0],
                },
            },
        ]
    }

    result = build_temporal_tracks_from_refined(payload, start=0, num_frames=2, max_cam_t_jump=3.0)

    assert result["tracks"]["ego_right"][1] is None
    assert result["frames"][1]["dropped_hands"]["ego_right"]["drop_reasons"] == ["large_cam_t_jump"]


def test_refined_temporal_keeps_low_keypoint_quality_hands():
    low_quality = {
        **hand(2, [20, 0, 30, 10], side="right"),
        "confidence": 0.8,
        "detector_score": 0.98,
        "keypoint_mean_conf": 0.58,
        "valid_keypoints": 8,
    }
    payload = {
        "frames": [
            {
                "frame_idx": 0,
                "ego_left_refined": hand(1, [0, 0, 10, 10], side="left"),
                "ego_right_refined": low_quality,
            },
        ]
    }

    result = build_temporal_tracks_from_refined(payload, start=0, num_frames=1)

    assert result["tracks"]["ego_left"][0] is not None
    assert result["tracks"]["ego_right"][0] is not None
    assert result["tracks"]["ego_right"][0]["keypoint_mean_conf"] == 0.58
    assert result["tracks"]["ego_right"][0]["valid_keypoints"] == 8
    assert result["frames"][0]["dropped_hands"] == {}


def test_refined_temporal_drops_large_mano_pose_jump_with_stable_bbox():
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_left_refined": hand_with_pose(1, [0, 0, 10, 10], 0.0)},
            {"frame_idx": 1, "ego_left_refined": hand_with_pose(2, [1, 0, 11, 10], 5.0)},
            {"frame_idx": 2, "ego_left_refined": hand_with_pose(3, [2, 0, 12, 10], 0.1)},
        ]
    }

    result = build_temporal_tracks_from_refined(
        payload,
        start=0,
        num_frames=3,
        max_mano_pose_jump=2.0,
        local_window_radius=1,
    )

    assert result["tracks"]["ego_left"][1] is None
    dropped = result["frames"][1]["dropped_hands"]["ego_left"]
    assert dropped["drop_reasons"] == ["large_mano_pose_jump"]
    assert dropped["mano_pose_local_deviation"] > 2.0
    assert result["tracks"]["ego_left"][2] is not None


def test_refined_temporal_drops_large_mesh_centroid_jump_with_stable_bbox(tmp_path: Path):
    first = hand(1, [0, 0, 10, 10])
    second = hand(2, [1, 0, 11, 10])
    third = hand(3, [2, 0, 12, 10])
    first["mesh_obj"] = write_mesh(tmp_path / "first.obj", [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    second["mesh_obj"] = write_mesh(tmp_path / "second.obj", [[10, 0, 0], [11, 0, 0], [10, 1, 0]])
    third["mesh_obj"] = write_mesh(tmp_path / "third.obj", [[0.1, 0, 0], [1.1, 0, 0], [0.1, 1, 0]])
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_left_refined": first},
            {"frame_idx": 1, "ego_left_refined": second},
            {"frame_idx": 2, "ego_left_refined": third},
        ]
    }

    result = build_temporal_tracks_from_refined(
        payload,
        start=0,
        num_frames=3,
        max_mesh_centroid_jump=2.0,
        local_window_radius=1,
    )

    assert result["tracks"]["ego_left"][1] is None
    dropped = result["frames"][1]["dropped_hands"]["ego_left"]
    assert dropped["drop_reasons"] == ["large_mesh_centroid_jump"]
    assert dropped["mesh_centroid_local_deviation"] > 2.0
    assert result["tracks"]["ego_left"][2] is not None


def test_refined_temporal_drops_large_mesh_vertex_rmse_with_stable_centroid(tmp_path: Path):
    first = hand(1, [0, 0, 10, 10])
    second = hand(2, [1, 0, 11, 10])
    third = hand(3, [2, 0, 12, 10])
    first["mesh_obj"] = write_mesh(tmp_path / "first.obj", [[-1, 0, 0], [1, 0, 0], [0, 0, 0]])
    second["mesh_obj"] = write_mesh(tmp_path / "second.obj", [[-5, 0, 0], [5, 0, 0], [0, 0, 0]])
    third["mesh_obj"] = write_mesh(tmp_path / "third.obj", [[-1.1, 0, 0], [1.1, 0, 0], [0, 0, 0]])
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_left_refined": first},
            {"frame_idx": 1, "ego_left_refined": second},
            {"frame_idx": 2, "ego_left_refined": third},
        ]
    }

    result = build_temporal_tracks_from_refined(
        payload,
        start=0,
        num_frames=3,
        max_mesh_vertex_rmse=1.0,
        local_window_radius=1,
    )

    assert result["tracks"]["ego_left"][1] is None
    dropped = result["frames"][1]["dropped_hands"]["ego_left"]
    assert dropped["drop_reasons"] == ["large_mesh_vertex_rmse"]
    assert dropped["mesh_vertex_rmse_local"] > 1.0
    assert result["tracks"]["ego_left"][2] is not None
