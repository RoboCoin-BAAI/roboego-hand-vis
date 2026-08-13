from pathlib import Path

from roboego_hand_vis.export.session_final_data import (
    clip_stage1_frames,
    final_segment_name,
    make_window_manifest,
    resolve_hand_visual_source,
)


def test_clip_stage1_frames_uses_inclusive_second_range():
    payload = {
        "source_video": "left.mp4",
        "frames": [
            {"t": 8, "activity": "before"},
            {"t": 9, "activity": "start"},
            {"t": 10, "activity": "middle"},
            {"t": 15, "activity": "end"},
            {"t": 16, "activity": "after"},
        ],
    }

    clipped = clip_stage1_frames(payload, start_s=9, end_s=15)

    assert [frame["t"] for frame in clipped["frames"]] == [9, 10, 15]
    assert clipped["source_range_s"] == {"start": 9, "end": 15}
    assert clipped["frame_count"] == 3


def test_make_window_manifest_maps_segment_and_local_frames():
    window = {
        "window_id": "handwin_0000",
        "window_type": "two_hand",
        "source_segment_id": "seg_0000",
        "start": 9,
        "end": 15,
        "core_range": {"start": 11, "end": 13},
        "segment_frame_range": {"start": 120, "end": 329, "num_frames": 210},
        "objects": ["product", "phone"],
        "actions": ["hold", "operate"],
    }

    manifest = make_window_manifest(window, fps=30.0)

    assert manifest["window_id"] == "handwin_0000"
    assert manifest["frames"]["segment_start"] == 120
    assert manifest["frames"]["segment_end"] == 329
    assert manifest["frames"]["local_start"] == 0
    assert manifest["frames"]["local_end"] == 209
    assert manifest["time_s"] == {"start": 9, "end": 15}


def test_final_segment_name_keeps_existing_segment_id_and_slug():
    path = Path("seg_0000_t000005_000039_operate_place_point_touch")

    assert final_segment_name(path) == "seg_0000_t000005_000039_operate_place_point_touch"


def test_resolve_hand_visual_source_prefers_temporal_size_filter(tmp_path):
    hamer_dir = tmp_path / "handwin_0000"
    overlay = hamer_dir / "hamer_overlay"
    temporal = hamer_dir / "refined_temporal_size_extreme_filter" / "temporal_annotated"
    overlay.mkdir(parents=True)
    temporal.mkdir(parents=True)
    (overlay / "frame_000120_hamer_overlay.jpg").write_bytes(b"overlay")
    (temporal / "frame_000120_temporal.jpg").write_bytes(b"temporal")

    directory, pattern, label = resolve_hand_visual_source(hamer_dir)

    assert directory == temporal
    assert pattern == "frame_{idx:06d}_temporal.jpg"
    assert label == "refined_temporal_size_extreme_filter"
