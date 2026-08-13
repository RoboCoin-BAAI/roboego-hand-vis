import pytest

from roboego_hand_vis.egohand.temporal_tracking import (
    apply_recovered_depth_overrides,
    build_side_tracks,
    build_recovered_depth_overrides_from_frames,
    apply_identity_and_jump_gate,
    filter_unstable_recovered_candidates,
    fill_edge_gaps,
    interpolate_short_gaps,
    process_tracks,
    smooth_track_translations,
)
from roboego_hand_vis.egohand.trajectory_smoothing import (
    clamp_rotation,
    kabsch_rotation,
    smooth_centers_weighted,
)


def obs(frame_idx, side, tx, status="detected"):
    return {
        "frame_idx": frame_idx,
        "side": side,
        "status": status,
        "translation": [tx, 0.0, 0.5],
        "source_anchor": [0.0, 0.0, 0.0],
        "target_anchor": [tx, 0.0, 0.5],
        "bbox_xyxy": [tx * 10.0, 10.0, tx * 10.0 + 5.0, 20.0],
        "source": "detected",
        "mesh_to_depth_median_m": 0.01,
        "valid_depth_ratio": 1.0,
    }


def test_interpolates_short_gap_and_marks_source():
    track = [obs(0, "ego_left", 0.0), None, obs(2, "ego_left", 2.0)]

    filled = interpolate_short_gaps(track, max_gap=2)

    assert filled[1] is not None
    assert filled[1]["status"] == "interpolated"
    assert filled[1]["source"] == "interpolated_between_neighbors"
    assert filled[1]["translation"] == pytest.approx([1.0, 0.0, 0.5])
    assert filled[1]["target_anchor"] == pytest.approx([1.0, 0.0, 0.5])
    assert filled[1]["bbox_xyxy"] == pytest.approx([10.0, 10.0, 15.0, 20.0])


def test_interpolated_gap_keeps_source_anchor_from_selected_mesh():
    previous = {
        **obs(0, "ego_left", 0.0),
        "mesh_obj": "previous.obj",
        "source_anchor": [10.0, 0.0, 0.0],
        "target_anchor": [20.0, 0.0, 0.5],
        "translation": [10.0, 0.0, 0.5],
    }
    following = {
        **obs(2, "ego_left", 2.0),
        "mesh_obj": "following.obj",
        "source_anchor": [30.0, 0.0, 0.0],
        "target_anchor": [40.0, 0.0, 0.5],
        "translation": [10.0, 0.0, 0.5],
    }

    filled = interpolate_short_gaps([previous, None, following], max_gap=2)

    assert filled[1]["mesh_obj"] == "previous.obj"
    assert filled[1]["interpolation"]["shape_source_frame_idx"] == 0
    assert filled[1]["target_anchor"] == pytest.approx([30.0, 0.0, 0.5])
    assert filled[1]["source_anchor"] == pytest.approx([10.0, 0.0, 0.0])
    assert filled[1]["translation"] == pytest.approx([20.0, 0.0, 0.5])


def test_does_not_interpolate_long_gap():
    track = [obs(0, "ego_left", 0.0), None, None, None, obs(4, "ego_left", 4.0)]

    filled = interpolate_short_gaps(track, max_gap=2)

    assert filled[1] is None
    assert filled[2] is None
    assert filled[3] is None


def test_fill_edge_gaps_uses_nearest_observation_for_short_boundary_gap():
    track = [None, obs(1, "ego_left", 1.0), obs(2, "ego_left", 2.0), None]

    filled = fill_edge_gaps(track, max_gap=1)

    assert filled[0] is not None
    assert filled[0]["status"] == "interpolated"
    assert filled[0]["source"] == "interpolated_from_edge_neighbor"
    assert filled[0]["frame_idx"] == 0
    assert filled[0]["target_anchor"] == pytest.approx([1.0, 0.0, 0.5])
    assert filled[3] is not None
    assert filled[3]["frame_idx"] == 3
    assert filled[3]["target_anchor"] == pytest.approx([2.0, 0.0, 0.5])


def test_smooth_track_translations_flags_large_raw_jump_without_moving_anchor():
    track = [obs(0, "ego_right", 0.0), obs(1, "ego_right", 1.0)]

    smoothed = smooth_track_translations(track, alpha=0.5, jump_threshold_m=0.5)

    assert smoothed[0]["smoothed_translation"] == pytest.approx([0.0, 0.0, 0.5])
    assert smoothed[1]["smoothed_translation"] == pytest.approx([1.0, 0.0, 0.5])
    assert smoothed[1]["smoothed_target_anchor"] == pytest.approx([1.0, 0.0, 0.5])
    assert "large_translation_jump" in smoothed[1]["anomaly_flags"]


def test_smooth_track_translations_keeps_interpolated_gap_fill_at_raw_interpolation():
    filled = interpolate_short_gaps(
        [obs(0, "ego_right", 0.0), None, obs(2, "ego_right", 2.0)],
        max_gap=2,
    )

    smoothed = smooth_track_translations(filled, alpha=0.1, jump_threshold_m=0.5)

    assert smoothed[1]["status"] == "interpolated"
    assert smoothed[1]["target_anchor"] == pytest.approx([1.0, 0.0, 0.5])
    assert smoothed[1]["smoothed_target_anchor"] == pytest.approx([1.0, 0.0, 0.5])
    assert smoothed[1]["smoothed_translation"] == pytest.approx([1.0, 0.0, 0.5])


def test_build_side_tracks_keeps_recovered_as_observation():
    report_frames = [
        {
            "frame_idx": 0,
            "hands": [
                {
                    "canonical_status": "ego_left",
                    "is_recovered": True,
                    "bbox_xyxy": [0, 0, 10, 10],
                    "mesh_obj": "left.obj",
                    "shape_preserved_fit_transform": {"translation": [0.0, 0.0, 0.5]},
                    "shape_preserved_mesh_to_depth_error": {"median_m": 0.01},
                    "shape_preserved_depth_to_mesh_error": {"median_m": 0.02},
                }
            ],
        }
    ]

    tracks = build_side_tracks(report_frames, start=0, num_frames=1)

    assert tracks["ego_left"][0]["status"] == "recovered"
    assert tracks["ego_right"][0] is None


def test_filters_isolated_low_area_recovered_candidate_with_large_temporal_jump():
    track = [
        obs(0, "ego_right", 0.0),
        {
            **obs(1, "ego_right", 2.0, status="recovered"),
            "bbox_area_ratio": 0.0005,
        },
        obs(2, "ego_right", 0.1),
    ]

    filtered = filter_unstable_recovered_candidates(
        track,
        jump_threshold_m=0.5,
        max_area_ratio=0.001,
    )

    assert filtered[0] is not None
    assert filtered[1] is None
    assert filtered[2] is not None


def test_keeps_low_area_recovered_candidate_when_temporally_continuous():
    track = [
        obs(0, "ego_right", 0.0),
        {
            **obs(1, "ego_right", 0.1, status="recovered"),
            "bbox_area_ratio": 0.0005,
        },
        obs(2, "ego_right", 0.2),
    ]

    filtered = filter_unstable_recovered_candidates(
        track,
        jump_threshold_m=0.5,
        max_area_ratio=0.001,
    )

    assert filtered[1] is not None
    assert filtered[1]["status"] == "recovered"


def hand_frame(frame_idx, side, z, is_recovered=False):
    source_anchor = [0.0, 0.0, 10.0]
    target_anchor = [0.0, 0.0, z]
    return {
        "frame_idx": frame_idx,
        "hands": [
            {
                "canonical_status": side,
                "is_recovered": is_recovered,
                "bbox_xyxy": [0, 0, 20, 20],
                "mesh_obj": f"{side}_{frame_idx}.obj",
                "shape_preserved_fit_transform": {
                    "source_anchor": source_anchor,
                    "target_anchor": target_anchor,
                    "target_z_median": z,
                    "translation": [0.0, 0.0, z - source_anchor[2]],
                },
                "shape_preserved_mesh_to_depth_error": {"median_m": 0.01},
            }
        ],
    }


def test_recovered_depth_overrides_use_neighboring_normal_z_before_smoothing():
    frames = [
        hand_frame(0, "ego_right", 0.50),
        hand_frame(1, "ego_right", 0.80, is_recovered=True),
        hand_frame(2, "ego_right", 0.60),
    ]

    overrides = build_recovered_depth_overrides_from_frames(frames)
    corrected = apply_recovered_depth_overrides(frames, overrides)
    tracks = process_tracks(corrected, start=0, num_frames=3, smooth_alpha=1.0)

    recovered = tracks["ego_right"][1]
    assert overrides[(1, "ego_right")]["target_z_m"] == pytest.approx(0.55)
    assert recovered["status"] == "recovered"
    assert recovered["target_anchor"][2] == pytest.approx(0.55)
    assert recovered["translation"][2] == pytest.approx(-9.45)
    assert recovered["smoothed_target_anchor"][2] == pytest.approx(0.55)


def test_identity_gate_swaps_crossed_left_right_observations():
    tracks = {
        "ego_left": [
            obs(0, "ego_left", 0.0),
            obs(1, "ego_left", 1.0),
        ],
        "ego_right": [
            obs(0, "ego_right", 1.0),
            obs(1, "ego_right", 0.0),
        ],
    }

    gated = apply_identity_and_jump_gate(
        tracks,
        enable_lr_identity_gate=True,
        drop_large_jumps=False,
        large_jump_threshold_m=0.5,
        swap_margin_m=0.05,
    )

    assert gated["ego_left"][1]["target_anchor"] == pytest.approx([0.0, 0.0, 0.5])
    assert gated["ego_left"][1]["side"] == "ego_left"
    assert "lr_identity_swapped" in gated["ego_left"][1]["anomaly_flags"]
    assert gated["ego_right"][1]["target_anchor"] == pytest.approx([1.0, 0.0, 0.5])
    assert gated["ego_right"][1]["side"] == "ego_right"


def test_large_jump_gate_drops_unexplained_jump_before_interpolation():
    frames = [
        hand_frame(0, "ego_left", 0.50),
        hand_frame(1, "ego_left", 1.20),
        hand_frame(2, "ego_left", 0.60),
    ]

    tracks = process_tracks(
        frames,
        start=0,
        num_frames=3,
        max_interp_gap=2,
        smooth_alpha=1.0,
        drop_large_jumps=True,
        large_jump_threshold_m=0.25,
    )

    assert tracks["ego_left"][1]["status"] == "interpolated"
    assert tracks["ego_left"][1]["target_anchor"][2] == pytest.approx(0.55)


def test_large_jump_gate_can_drop_large_bbox_center_jump_even_when_3d_jump_is_small():
    first = obs(0, "ego_left", 0.0)
    jumped = {
        **obs(1, "ego_left", 0.08),
        "bbox_xyxy": [500.0, 10.0, 550.0, 60.0],
    }
    last = obs(2, "ego_left", 0.1)
    tracks = {
        "ego_left": [first, jumped, last],
        "ego_right": [None, None, None],
    }

    gated = apply_identity_and_jump_gate(
        tracks,
        enable_lr_identity_gate=False,
        drop_large_jumps=True,
        large_jump_threshold_m=0.12,
        max_bbox_center_jump_px=200.0,
    )

    assert gated["ego_left"][1] is None


def test_weighted_center_smoothing_trusts_normal_more_than_recovered():
    centers = [
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
    ]

    smoothed = smooth_centers_weighted(
        centers,
        statuses=["normal", "recovered", "normal"],
        window=3,
        weights={"normal": 1.0, "recovered": 0.1},
        max_corrections_m={"normal": 0.015, "recovered": 0.05},
    )

    assert smoothed[0]["center"] == pytest.approx([0.015, 0.0, 0.5])
    assert smoothed[1]["center"][0] == pytest.approx(1.0)
    assert smoothed[2]["center"] == pytest.approx([1.985, 0.0, 0.5])
    assert smoothed[0]["correction_norm_m"] == pytest.approx(0.015)


def test_kabsch_rotation_and_clamp_limit_orientation_change():
    source = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    target = [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    rotation, angle = kabsch_rotation(source, target)
    clamped, clamped_angle = clamp_rotation(rotation, max_degrees=10.0)

    assert angle == pytest.approx(90.0)
    assert clamped_angle == pytest.approx(10.0)
    rotated = source[0] @ clamped.T
    assert rotated[0] == pytest.approx(0.9848077)
    assert rotated[1] == pytest.approx(0.1736482)
