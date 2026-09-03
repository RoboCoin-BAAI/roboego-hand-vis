from __future__ import annotations

import numpy as np

from roboego_hand_vis.privacy.egoblur_temporal import (
    Detection,
    FaceBlurConfig,
    build_temporal_tracks,
    detections_to_frame_boxes,
    expand_detection_schedule,
    interpolate_tracks,
)


def test_expand_detection_schedule_samples_every_n_frames_and_last_frame():
    assert expand_detection_schedule(frame_count=10, detect_every_n_frames=3) == [0, 3, 6, 9]
    assert expand_detection_schedule(frame_count=11, detect_every_n_frames=3) == [0, 3, 6, 9, 10]


def test_temporal_tracks_interpolate_short_gaps_between_three_frame_detections():
    detections = {
        0: [Detection(bbox=[0, 0, 10, 10], score=0.9)],
        3: [Detection(bbox=[30, 0, 40, 10], score=0.8)],
        6: [Detection(bbox=[60, 0, 70, 10], score=0.7)],
    }

    tracks = build_temporal_tracks(
        detections,
        frame_count=7,
        config=FaceBlurConfig(detect_every_n_frames=3, max_gap_frames=3, smooth_window=1),
    )
    frame_boxes = detections_to_frame_boxes(tracks, frame_count=7)

    assert sorted(frame_boxes) == list(range(7))
    np.testing.assert_allclose(frame_boxes[1][0]["bbox"], [10, 0, 20, 10])
    np.testing.assert_allclose(frame_boxes[2][0]["bbox"], [20, 0, 30, 10])
    np.testing.assert_allclose(frame_boxes[4][0]["bbox"], [40, 0, 50, 10])
    np.testing.assert_allclose(frame_boxes[5][0]["bbox"], [50, 0, 60, 10])


def test_interpolate_tracks_does_not_fill_long_gaps():
    detections = {
        0: [Detection(bbox=[0, 0, 10, 10], score=0.9)],
        10: [Detection(bbox=[100, 0, 110, 10], score=0.8)],
    }

    tracks = interpolate_tracks(
        build_temporal_tracks(
            detections,
            frame_count=11,
            config=FaceBlurConfig(detect_every_n_frames=3, max_gap_frames=3, smooth_window=1),
        ),
        max_gap_frames=3,
        smooth_window=1,
    )
    frame_boxes = detections_to_frame_boxes(tracks, frame_count=11)

    assert 1 not in frame_boxes
    assert 9 not in frame_boxes
