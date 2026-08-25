import json
from pathlib import Path

from roboego_hand_vis.restoration.realvdeblur import build_deblur_manifest, compute_output_geometry


def base_manifest(tmp_path: Path) -> dict:
    left = tmp_path / "left_rectified.mp4"
    right = tmp_path / "right_rectified.mp4"
    left.write_bytes(b"fake")
    right.write_bytes(b"fake")
    return {
        "schema_version": "1.0",
        "dataset_name": "wumei01",
        "sequence_name": "handwin",
        "reference_camera": "left",
        "videos": {
            "left": {"path": str(left), "width": 1920, "height": 1074, "fps": 30.0, "frame_count": 240},
            "right": {"path": str(right), "width": 1920, "height": 1074, "fps": 30.0, "frame_count": 240},
        },
        "intrinsics": {
            "left": {"fx": 444.0, "fy": 444.0, "cx": 960.0, "cy": 544.0, "width": 1920, "height": 1074},
            "right": {"fx": 444.0, "fy": 444.0, "cx": 960.0, "cy": 544.0, "width": 1920, "height": 1074},
        },
        "extrinsics": {"baseline_m": 0.1},
    }


def test_compute_output_geometry_crops_to_multiple_of_16_by_default():
    geometry = compute_output_geometry(width=1920, height=1074, target_width=None, target_height=None)

    assert geometry.width == 1920
    assert geometry.height == 1072
    assert geometry.scale == 1.0
    assert geometry.crop_left == 0
    assert geometry.crop_top == 1


def test_build_deblur_manifest_updates_video_paths_and_cropped_intrinsics(tmp_path):
    manifest = base_manifest(tmp_path)
    output_dir = tmp_path / "deblur_realvdeblur"
    left_out = output_dir / "videos" / "left_rectified.mp4"
    right_out = output_dir / "videos" / "right_rectified.mp4"

    updated = build_deblur_manifest(
        manifest,
        output_dir=output_dir,
        left_video=left_out,
        right_video=right_out,
        width=1920,
        height=1072,
        fps=30.0,
        frame_count=240,
        geometry=compute_output_geometry(width=1920, height=1074, target_width=None, target_height=None),
    )

    assert updated["videos"]["left"]["path"] == str(left_out.resolve())
    assert updated["videos"]["left"]["height"] == 1072
    assert updated["intrinsics"]["left"]["cy"] == 543.0
    assert updated["intrinsics"]["right"]["height"] == 1072
    assert updated["restoration"]["method"] == "RealVDeblur"


def test_build_deblur_manifest_remains_json_serializable(tmp_path):
    payload = build_deblur_manifest(
        base_manifest(tmp_path),
        output_dir=tmp_path / "deblur",
        left_video=tmp_path / "deblur/videos/left.mp4",
        right_video=tmp_path / "deblur/videos/right.mp4",
        width=960,
        height=544,
        fps=30.0,
        frame_count=10,
        geometry=compute_output_geometry(width=1920, height=1074, target_width=960, target_height=544),
    )

    json.dumps(payload)
