from pathlib import Path

import cv2
import numpy as np
import pytest

from roboego_hand_vis.objects.sam3_backend import load_sam3_mask_observation


def test_load_sam3_mask_observation_reads_frame_mask(tmp_path: Path):
    masks_root = tmp_path / "video_segmentation/masks"
    frame_dir = masks_root / "frame_000005_masks"
    frame_dir.mkdir(parents=True)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[6:10, 8:15] = 255
    cv2.imwrite(str(frame_dir / "cup.png"), mask)
    obs, loaded = load_sam3_mask_observation(masks_root, frame_idx=5, object_id="cup")
    assert obs.bbox_xywh == [8, 6, 7, 4]
    assert obs.mask_area_px == 28
    assert loaded.shape == (20, 30)


def test_load_sam3_mask_observation_returns_empty_mask_for_missing_file(tmp_path: Path):
    obs, loaded = load_sam3_mask_observation(tmp_path, frame_idx=5, object_id="cup")

    assert obs.bbox_xywh is None
    assert obs.mask_area_px == 0
    assert obs.source == "sam3_mask_missing"
    assert loaded.shape == (0, 0)
    assert loaded.dtype == np.uint8


def test_load_sam3_mask_observation_preserves_present_empty_mask_shape(tmp_path: Path):
    frame_dir = tmp_path / "frame_000005_masks"
    frame_dir.mkdir(parents=True)
    cv2.imwrite(str(frame_dir / "cup.png"), np.zeros((12, 13), dtype=np.uint8))

    obs, loaded = load_sam3_mask_observation(tmp_path, frame_idx=5, object_id="cup")

    assert obs.bbox_xywh is None
    assert obs.mask_area_px == 0
    assert obs.source == "sam3_mask"
    assert loaded.shape == (12, 13)


def test_load_sam3_mask_observation_normalizes_nonzero_grayscale_values(tmp_path: Path):
    frame_dir = tmp_path / "frame_000005_masks"
    frame_dir.mkdir(parents=True)
    mask = np.array([[0, 1], [128, 0]], dtype=np.uint8)
    cv2.imwrite(str(frame_dir / "cup.png"), mask)

    _, loaded = load_sam3_mask_observation(tmp_path, frame_idx=5, object_id="cup")

    np.testing.assert_array_equal(loaded, np.array([[0, 255], [255, 0]], dtype=np.uint8))


def test_load_sam3_mask_observation_rejects_corrupt_existing_png(tmp_path: Path):
    frame_dir = tmp_path / "frame_000005_masks"
    frame_dir.mkdir(parents=True)
    (frame_dir / "cup.png").write_bytes(b"not a valid png")

    with pytest.raises(RuntimeError, match="Could not read mask"):
        load_sam3_mask_observation(tmp_path, frame_idx=5, object_id="cup")
