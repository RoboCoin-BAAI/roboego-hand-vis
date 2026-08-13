import sys
from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.datasets.crop_manifest import (
    crop_frame,
    crop_intrinsics,
    roi_from_ratios,
    update_manifest_for_crop,
)


def write_tiny_video(path: Path, width: int, height: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    assert writer.isOpened()
    writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def test_roi_from_ratios_converts_to_pixel_bounds():
    roi = roi_from_ratios(width=100, height=80, x_min=0.25, x_max=0.75, y_min=0.4, y_max=1.0)

    assert roi == (25, 32, 75, 80)


def test_crop_intrinsics_shifts_principal_point_and_updates_resolution():
    intrinsics = {"fx": 10.0, "fy": 11.0, "cx": 50.0, "cy": 40.0, "width": 100, "height": 80}

    cropped = crop_intrinsics(intrinsics, roi=(25, 32, 75, 80))

    assert cropped["fx"] == 10.0
    assert cropped["fy"] == 11.0
    assert cropped["cx"] == 25.0
    assert cropped["cy"] == 8.0
    assert cropped["width"] == 50
    assert cropped["height"] == 48


def test_crop_frame_uses_xyxy_bounds():
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    cropped = crop_frame(frame, roi=(1, 1, 5, 3))

    assert np.array_equal(cropped, frame[1:3, 1:5])


def test_update_manifest_for_crop_points_to_cropped_videos_and_records_roi(tmp_path):
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "wumei01",
        "sequence_name": "wumei01",
        "root_dir": str(tmp_path),
        "reference_camera": "left",
        "videos": {
            "left": {"path": str(tmp_path / "left.mp4"), "width": 100, "height": 80, "fps": 30.0},
            "right": {"path": str(tmp_path / "right.mp4"), "width": 100, "height": 80, "fps": 30.0},
        },
        "intrinsics": {
            "left": {"fx": 10.0, "fy": 11.0, "cx": 50.0, "cy": 40.0, "width": 100, "height": 80},
            "right": {"fx": 10.0, "fy": 11.0, "cx": 50.0, "cy": 40.0, "width": 100, "height": 80},
        },
        "extrinsics": {"baseline_m": 0.1},
        "camera_model": {"type": "pinhole", "is_rectified": True, "is_undistorted": True},
        "timestamps": {},
        "trajectory": {},
        "annotations": {},
        "metadata": {},
    }
    write_tiny_video(tmp_path / "crop_left.mp4", width=50, height=48)
    write_tiny_video(tmp_path / "crop_right.mp4", width=50, height=48)

    out = update_manifest_for_crop(
        manifest,
        left_video=tmp_path / "crop_left.mp4",
        right_video=tmp_path / "crop_right.mp4",
        roi=(25, 32, 75, 80),
    )

    assert out["videos"]["left"]["path"] == str(tmp_path / "crop_left.mp4")
    assert out["intrinsics"]["left"]["cx"] == 25.0
    assert out["intrinsics"]["left"]["cy"] == 8.0
    assert out["intrinsics"]["left"]["width"] == 50
    assert out["metadata"]["crop"]["roi_xyxy"] == [25, 32, 75, 80]
