import cv2
import numpy as np

from roboego_hand_vis.objects.color_yellow import (
    YellowSegmentationConfig,
    detect_yellow_object,
    fill_missing_observations,
    save_mask_and_crop,
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


def test_fill_missing_observations_fills_exactly_max_gap():
    frames = [
        ObjectObservation(0, "yellow_object_0", [0, 0, 10, 10], 100, "detected"),
        *[
            ObjectObservation(i, "yellow_object_0", None, 0, "missing")
            for i in range(1, 16)
        ],
        ObjectObservation(16, "yellow_object_0", [16, 16, 10, 10], 100, "detected"),
    ]

    filled = fill_missing_observations(frames, max_gap=15)

    assert all(obs.bbox_xywh is not None for obs in filled[1:16])
    assert filled[8].bbox_xywh == [8, 8, 10, 10]
    assert all(obs.source == "interpolated_yellow_bbox" for obs in filled[1:16])


def test_fill_missing_observations_leaves_long_gap_missing():
    frames = [
        ObjectObservation(0, "yellow_object_0", [0, 0, 10, 10], 100, "detected"),
        *[
            ObjectObservation(i, "yellow_object_0", None, 0, "missing")
            for i in range(1, 18)
        ],
        ObjectObservation(18, "yellow_object_0", [18, 18, 10, 10], 100, "detected"),
    ]

    filled = fill_missing_observations(frames, max_gap=15)

    assert all(obs.bbox_xywh is None for obs in filled[1:18])
    assert all(obs.source == "missing" for obs in filled[1:18])


def test_detect_yellow_object_returns_missing_when_lower_roi_has_no_yellow():
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    obs, mask = detect_yellow_object(
        image,
        frame_idx=3,
        object_id="yellow_object_0",
        config=YellowSegmentationConfig(min_area_px=20),
    )

    assert obs.bbox_xywh is None
    assert obs.mask_area_px == 0
    assert obs.source == "missing"
    assert mask.sum() == 0


def test_save_mask_and_crop_writes_artifacts(tmp_path):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[4:12, 5:15] = [255, 230, 20]
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:12, 5:15] = 255
    obs = ObjectObservation(3, "yellow_object_0", [5, 4, 10, 8], 80, "detected")

    saved = save_mask_and_crop(
        image_rgb=image,
        mask=mask,
        obs=obs,
        output_dir=tmp_path,
        crop_padding_px=2,
    )

    assert saved.mask_path is not None
    assert saved.crop_path is not None
    assert saved.mask_path.parent == tmp_path / "masks"
    assert saved.crop_path.parent == tmp_path / "crops"
    assert saved.mask_path.exists()
    assert saved.crop_path.exists()
    assert cv2.imread(str(saved.crop_path)) is not None
