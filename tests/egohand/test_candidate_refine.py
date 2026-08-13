from roboego_hand_vis.egohand.candidate_refine import CandidateRefineConfig, refine_candidates_payload


def cand(candidate_id: int, side: str, bbox: list[float]) -> dict:
    return {"candidate_id": candidate_id, "side": side, "bbox_xyxy": bbox, "image_width": 1000, "image_height": 1000}


def strict_config(**kwargs):
    return CandidateRefineConfig(refine_mode="strict_ego_audit", **kwargs)


def legacy_config(**kwargs):
    return CandidateRefineConfig(refine_mode="legacy_score", **kwargs)


def test_strict_refine_uses_hands_visible_and_hamer_anatomical_side():
    payload = {
        "frames": [
            {
                "frame_idx": 0,
                "candidates": [
                    cand(1, "left", [300, 650, 500, 940]),
                    cand(2, "right", [600, 650, 800, 940]),
                ],
            }
        ]
    }
    stage1 = {"frames": [{"t": 0, "hands_visible": {"left": True, "right": False}}]}

    refined = refine_candidates_payload(payload, stage1, config=strict_config())

    frame = refined["frames"][0]
    assert frame["refine"]["visibility_state"] == "only_left"
    assert frame["ego_left_refined"]["candidate_id"] == 1
    assert frame["ego_right_refined"] is None
    assert frame["missing_hands"] == []
    assert frame["other_hands"][0]["candidate_id"] == 2
    assert "semantic_side_not_visible" in frame["other_hands"][0]["refine_reasons"]


def test_strict_refine_marks_expected_hand_missing_without_forcing_other_side():
    payload = {"frames": [{"frame_idx": 0, "candidates": [cand(1, "right", [600, 650, 800, 940])]}]}
    stage1 = {"frames": [{"t": 0, "hands_visible": {"left": True, "right": False}}]}

    refined = refine_candidates_payload(payload, stage1, config=strict_config())

    frame = refined["frames"][0]
    assert frame["refine"]["visibility_state"] == "none"
    assert frame["ego_left_refined"] is None
    assert frame["ego_right_refined"] is None
    assert frame["missing_hands"] == [{"side": "left", "reason": "stage1_expected_but_no_reliable_candidate"}]
    assert frame["other_hands"][0]["candidate_id"] == 1


def test_strict_refine_marks_top_or_edge_hands_as_other_even_if_semantic_expected():
    payload = {
        "frames": [
            {
                "frame_idx": 0,
                "candidates": [
                    cand(1, "left", [300, 80, 500, 220]),
                    cand(2, "right", [930, 650, 995, 900]),
                ],
            }
        ]
    }
    stage1 = {"frames": [{"t": 0, "hands_visible": {"left": True, "right": True}}]}

    refined = refine_candidates_payload(payload, stage1, config=strict_config(min_center_x_ratio=0.1, max_center_x_ratio=0.9))

    frame = refined["frames"][0]
    assert frame["refine"]["visibility_state"] == "none"
    assert {item["candidate_id"] for item in frame["other_hands"]} == {1, 2}
    reasons_by_id = {item["candidate_id"]: item["refine_reasons"] for item in frame["other_hands"]}
    assert "not_from_wearer_lower_body" in reasons_by_id[1]
    assert "enters_from_side_edge" in reasons_by_id[2]


def test_refine_candidates_accepts_single_ego_hand_without_forcing_the_other_side():
    payload = {"frames": [{"frame_idx": 0, "candidates": [cand(1, "left", [300, 500, 550, 900])]}]}
    refined = refine_candidates_payload(payload, {"frames": [{"t": 0, "hands_visible": {"left": True, "right": False}}]}, config=legacy_config())

    frame = refined["frames"][0]
    assert frame["refine"]["visibility_state"] == "only_left"
    assert frame["ego_left_refined"]["candidate_id"] == 1
    assert frame["ego_right_refined"] is None


def test_refine_candidates_marks_large_temporal_jump_uncertain():
    payload = {
        "frames": [
            {"frame_idx": 0, "candidates": [cand(1, "right", [300, 500, 550, 900])]},
            {"frame_idx": 1, "candidates": [cand(2, "right", [10, 20, 80, 100])]},
        ]
    }

    refined = refine_candidates_payload(payload, {}, config=legacy_config(max_center_jump_px=100))

    second = refined["frames"][1]["refined_candidates"][0]
    assert "large_center_jump" in second["refine_risks"]
    assert second["refine_role"] in {"uncertain", "rejected", "other"}


def test_accept_uncertain_refined_selects_overlay_confirmed_temporal_jump_candidate():
    payload = {
        "frames": [
            {"frame_idx": 0, "candidates": [cand(1, "right", [300, 500, 550, 900])]},
            {"frame_idx": 1, "candidates": [dict(cand(2, "right", [760, 640, 900, 900]), median_depth_m=0.6)]},
        ]
    }
    stage1 = {"frames": [{"t": 0, "activity": "wearer holds a product with right hand"}]}

    refined = refine_candidates_payload(
        payload,
        stage1,
        config=legacy_config(max_center_jump_px=100, accept_uncertain=True),
    )

    second = refined["frames"][1]
    assert second["refine"]["visibility_state"] == "only_right"
    assert second["ego_right_refined"]["candidate_id"] == 2
    assert second["ego_right_refined"]["refine_quality"] == "unstable"
    assert "large_center_jump" in second["ego_right_refined"]["refine_risks"]


def test_refine_candidates_uses_status_to_infer_side():
    payload = {"frames": [{"frame_idx": 0, "candidates": [dict(cand(1, "unknown", [300, 500, 550, 900]), status="ego_right", side=None)]}]}

    refined = refine_candidates_payload(payload, {}, config=legacy_config())

    frame = refined["frames"][0]
    assert frame["refine"]["visibility_state"] == "only_right"
    assert frame["ego_right_refined"]["candidate_id"] == 1


def test_refine_candidates_demotes_side_edge_candidate_without_temporal_support():
    payload = {"frames": [{"frame_idx": 0, "candidates": [dict(cand(1, "right", [20, 650, 130, 850]), status="ego_right")]}]}

    refined = refine_candidates_payload(payload, {}, config=legacy_config(min_center_x_ratio=0.2, side_edge_penalty=2.0))

    candidate = refined["frames"][0]["refined_candidates"][0]
    assert "near_side_edge" in candidate["refine_risks"]
    assert candidate["refine_role"] in {"other", "uncertain"}
    assert refined["frames"][0]["ego_right_refined"] is None


def test_permissive_stage1_single_hand_keeps_temporally_continuous_other_hand():
    payload = {
        "frames": [
            {
                "frame_idx": 0,
                "candidates": [
                    cand(1, "left", [300, 700, 420, 900]),
                    cand(2, "right", [600, 700, 720, 900]),
                ],
            },
            {
                "frame_idx": 1,
                "candidates": [
                    cand(3, "left", [305, 700, 425, 900]),
                    cand(4, "right", [605, 700, 725, 900]),
                ],
            },
        ]
    }
    stage1 = {"frames": [{"t": 0, "activity": "wearer holds a product with right hand"}]}

    refined = refine_candidates_payload(payload, stage1, config=legacy_config(stage1_hand_policy="permissive"))

    frame = refined["frames"][1]
    assert frame["refine"]["visibility_state"] == "both_hands"
    assert frame["ego_left_refined"]["candidate_id"] == 3
    assert frame["ego_right_refined"]["candidate_id"] == 4


def test_conservative_stage1_single_hand_overrides_to_named_hand_side():
    payload = {"frames": [{"frame_idx": 0, "candidates": [dict(cand(1, "left", [300, 700, 420, 900]), status="ego_left")]}]}
    stage1 = {"frames": [{"t": 0, "activity": "wearer holds a product with right hand"}]}

    refined = refine_candidates_payload(payload, stage1, config=legacy_config(stage1_hand_policy="conservative"))

    frame = refined["frames"][0]
    assert frame["refine"]["visibility_state"] == "only_right"
    assert frame["ego_right_refined"]["candidate_id"] == 1


def test_refine_candidates_maps_segment_local_frame_index_to_session_stage1_time():
    payload = {"frames": [{"frame_idx": 120, "candidates": [dict(cand(1, "left", [300, 700, 420, 900]), status="ego_left")]}]}
    stage1 = {
        "frames": [
            {"t": 4, "activity": "wearer holds a phone in the left hand"},
            {"t": 9, "activity": "wearer holds a product with right hand"},
        ]
    }

    refined = refine_candidates_payload(
        payload,
        stage1,
        config=legacy_config(fps=30, stage1_time_offset_s=5, stage1_hand_policy="conservative"),
    )

    frame = refined["frames"][0]
    assert frame["refine"]["stage1_t"] == 9
    assert frame["refine"]["visibility_state"] == "only_right"
    assert frame["ego_right_refined"]["candidate_id"] == 1
