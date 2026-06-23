import pytest

from roboegopipe.egohand.temporal_tracking import (
    build_side_tracks,
    filter_unstable_recovered_candidates,
    interpolate_short_gaps,
    smooth_track_translations,
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


def test_does_not_interpolate_long_gap():
    track = [obs(0, "ego_left", 0.0), None, None, None, obs(4, "ego_left", 4.0)]

    filled = interpolate_short_gaps(track, max_gap=2)

    assert filled[1] is None
    assert filled[2] is None
    assert filled[3] is None


def test_smooth_track_translations_flags_large_raw_jump():
    track = [obs(0, "ego_right", 0.0), obs(1, "ego_right", 1.0)]

    smoothed = smooth_track_translations(track, alpha=0.5, jump_threshold_m=0.5)

    assert smoothed[0]["smoothed_translation"] == pytest.approx([0.0, 0.0, 0.5])
    assert smoothed[1]["smoothed_translation"] == pytest.approx([0.5, 0.0, 0.5])
    assert "large_translation_jump" in smoothed[1]["anomaly_flags"]


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
