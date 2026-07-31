import numpy as np

from roboego_hand_vis.objects.color_yellow import (
    YellowSegmentationConfig,
    detect_yellow_object,
    fill_missing_observations,
)
from roboego_hand_vis.objects.segmentation import ObjectObservation


def test_detect_yellow_object_returns_largest_lower_roi_component():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[60:85, 40:80] = [255, 230, 20]
    image[5:20, 5:35] = [255, 230, 20]
    obs, mask = detect_yellow_object(
        image,
        frame_idx=3,
        object_id="yellow_object_0",
        config=YellowSegmentationConfig(roi_y_min_ratio=0.4, min_area_px=20),
    )
    assert obs.bbox_xywh == [40, 60, 40, 25]
    assert obs.mask_area_px == 1000
    assert mask.dtype == np.uint8


def test_fill_missing_observations_interpolates_short_gap():
    frames = [
        ObjectObservation(0, "yellow_object_0", [0, 0, 10, 10], 100, "detected"),
        ObjectObservation(1, "yellow_object_0", None, 0, "missing"),
        ObjectObservation(2, "yellow_object_0", [10, 10, 10, 10], 100, "detected"),
    ]
    filled = fill_missing_observations(frames, max_gap=3)
    assert filled[1].bbox_xywh == [5, 5, 10, 10]
    assert filled[1].source == "interpolated_yellow_bbox"
