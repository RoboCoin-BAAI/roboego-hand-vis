from roboego_hand_vis.egohand.refined_temporal import build_temporal_tracks_from_refined


def hand(candidate_id: int, bbox: list[float], side: str = "left") -> dict:
    return {
        "candidate_id": candidate_id,
        "bbox_xyxy": bbox,
        "handedness": side,
        "refine_quality": "reliable",
    }


def test_refined_temporal_keeps_reliable_refine_observations_as_normal():
    payload = {"frames": [{"frame_idx": 10, "ego_left_refined": hand(1, [10, 20, 30, 60])}]}

    result = build_temporal_tracks_from_refined(payload, start=10, num_frames=1)

    obs = result["tracks"]["ego_left"][0]
    assert obs["status"] == "normal"
    assert obs["source"] == "strict_refine_reliable"
    assert obs["candidate_id"] == 1
    assert result["summary"]["ego_left"]["normal"] == 1


def test_refined_temporal_interpolates_short_missing_gap_without_touching_neighbors():
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_left_refined": hand(1, [0, 0, 10, 10])},
            {"frame_idx": 2, "ego_left_refined": hand(2, [20, 0, 30, 10])},
        ]
    }

    result = build_temporal_tracks_from_refined(payload, start=0, num_frames=3, max_gap=2)

    track = result["tracks"]["ego_left"]
    assert track[0]["status"] == "normal"
    assert track[1]["status"] == "interpolated"
    assert track[1]["bbox_xyxy"] == [10.0, 0.0, 20.0, 10.0]
    assert track[2]["status"] == "normal"


def test_refined_temporal_flags_large_bbox_jump_but_keeps_refine_result():
    payload = {
        "frames": [
            {"frame_idx": 0, "ego_right_refined": hand(1, [0, 0, 10, 10], side="right")},
            {"frame_idx": 1, "ego_right_refined": hand(2, [500, 0, 510, 10], side="right")},
        ]
    }

    result = build_temporal_tracks_from_refined(payload, start=0, num_frames=2, max_bbox_jump_px=100)

    obs = result["tracks"]["ego_right"][1]
    assert obs["status"] == "normal"
    assert "large_bbox_center_jump" in obs["anomaly_flags"]
    assert result["summary"]["ego_right"]["large_bbox_center_jump"] == 1

