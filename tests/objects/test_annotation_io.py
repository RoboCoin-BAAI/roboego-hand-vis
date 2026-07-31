from pathlib import Path

import numpy as np

from roboego_hand_vis.objects.annotation_io import write_object_tracks_json
from roboego_hand_vis.objects.segmentation import ObjectObservation, bbox_xywh_to_xyxy, crop_with_padding


def test_bbox_xywh_to_xyxy():
    assert bbox_xywh_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]


def test_crop_with_padding_clips_to_image_bounds():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = crop_with_padding(image, [25, 15, 20, 20], padding_px=4)
    assert crop.shape == (9, 9, 3)


def test_write_object_tracks_json_uses_relative_artifact_paths(tmp_path: Path):
    obs = ObjectObservation(
        frame_idx=0,
        object_id="yellow_object_0",
        bbox_xywh=[1, 2, 3, 4],
        mask_area_px=12,
        source="test",
        mask_path=tmp_path / "masks/frame_000000_yellow_object_0.png",
        crop_path=tmp_path / "crops/frame_000000_yellow_object_0.png",
        depth_median_m=0.35,
        depth_valid_ratio=0.9,
    )
    out = tmp_path / "object_tracks.json"
    payload = write_object_tracks_json(
        observations={"yellow_object_0": [obs]},
        output_path=out,
        output_dir=tmp_path,
        backend="color_yellow",
        video_path=Path("video.mp4"),
        depth_dir=Path("depth"),
    )
    frame = payload["objects"]["yellow_object_0"]["frames"][0]
    assert frame["mask_path"] == "masks/frame_000000_yellow_object_0.png"
    assert frame["crop_path"] == "crops/frame_000000_yellow_object_0.png"
    assert out.exists()
