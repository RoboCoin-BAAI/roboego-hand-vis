from pathlib import Path

import cv2
import numpy as np

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
