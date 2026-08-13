import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.datasets.manifest import ResolveConfig, resolve_dataset_manifest, write_manifest


def write_tiny_video(path: Path, width: int = 4, height: int = 4, frames: int = 2, fps: float = 5.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for idx in range(frames):
        frame = np.full((height, width, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def write_intrinsics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "left": {
                    "fx": 10.0,
                    "fy": 11.0,
                    "cx": 2.0,
                    "cy": 1.5,
                    "distortion_model": None,
                    "distortion": [],
                    "extrinsics": np.eye(4).tolist(),
                },
                "right": {
                    "fx": 10.5,
                    "fy": 11.5,
                    "cx": 2.1,
                    "cy": 1.6,
                    "distortion_model": None,
                    "distortion": [],
                    "extrinsics": np.eye(4).tolist(),
                },
                "baseline_m": 0.064,
            }
        ),
        encoding="utf-8",
    )


def write_wumei01_rectify_yaml(path: Path) -> None:
    path.write_text(
        """
rectified:
  image_size: [1920, 1074]
  K1_rect:
    - [444.4, 0.0, 959.9]
    - [0.0, 444.4, 543.9]
    - [0.0, 0.0, 1.0]
  K2_rect:
    - [444.4, 0.0, 959.9]
    - [0.0, 444.4, 543.9]
    - [0.0, 0.0, 1.0]
  P2:
    - [444.4, 0.0, 959.9, -44.35]
    - [0.0, 444.4, 543.9, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
stereo_params:
  baseline_m: -0.0998
source_model:
  type: equidistant
""",
        encoding="utf-8",
    )


def test_generic_stereo_manifest_contains_standard_paths_and_camera_fields(tmp_path):
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    intrinsics = tmp_path / "camera_intrinsics.json"
    write_tiny_video(left)
    write_tiny_video(right)
    write_intrinsics(intrinsics)

    manifest = resolve_dataset_manifest(
        ResolveConfig(
            dataset="generic_stereo",
            root=tmp_path,
            output_dir=tmp_path / "manifest",
            left_video=left,
            right_video=right,
            intrinsics_json=intrinsics,
        )
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["reference_camera"] == "left"
    assert manifest["videos"]["left"]["width"] == 4
    assert manifest["videos"]["right"]["height"] == 4
    assert manifest["intrinsics"]["left"]["fx"] == 10.0
    assert manifest["intrinsics"]["right"]["fy"] == 11.5
    assert manifest["extrinsics"]["baseline_m"] == 0.064
    assert manifest["extrinsics"]["left_to_right"]["t"] == [0.064, 0.0, 0.0]


def test_write_manifest_outputs_json_and_summary(tmp_path):
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    intrinsics = tmp_path / "camera_intrinsics.json"
    write_tiny_video(left)
    write_tiny_video(right)
    write_intrinsics(intrinsics)

    config = ResolveConfig(
        dataset="generic_stereo",
        root=tmp_path,
        output_dir=tmp_path / "manifest",
        left_video=left,
        right_video=right,
        intrinsics_json=intrinsics,
    )
    manifest = resolve_dataset_manifest(config)
    manifest_path, summary_path = write_manifest(manifest, config.output_dir)

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")
    assert saved["dataset_name"] == "generic_stereo"
    assert "left_video:" in summary
    assert "baseline_m: 0.064" in summary


def test_wumei01_manifest_uses_rectified_videos_and_positive_baseline(tmp_path):
    videos = tmp_path / "videos"
    timestamps = tmp_path / "timestamps"
    calibrations = tmp_path / "calibrations"
    videos.mkdir()
    timestamps.mkdir()
    calibrations.mkdir()
    left = videos / "left_rectified.mp4"
    right = videos / "right_rectified.mp4"
    write_tiny_video(left, width=8, height=6)
    write_tiny_video(right, width=8, height=6)
    (timestamps / "left_timestamps.txt").write_text("0.0\n", encoding="utf-8")
    (timestamps / "right_timestamps.txt").write_text("0.0\n", encoding="utf-8")
    write_wumei01_rectify_yaml(calibrations / "rectify_params.yaml")

    manifest = resolve_dataset_manifest(
        ResolveConfig(
            dataset="wumei01",
            root=tmp_path,
            output_dir=tmp_path / "manifest",
            sequence_name="wumei01",
        )
    )

    assert manifest["dataset_name"] == "wumei01"
    assert manifest["videos"]["left"]["path"].endswith("videos/left_rectified.mp4")
    assert manifest["intrinsics"]["left"]["fx"] == 444.4
    assert manifest["intrinsics"]["right"]["cy"] == 543.9
    assert manifest["extrinsics"]["baseline_m"] == 0.0998
    assert manifest["extrinsics"]["left_to_right"]["t"] == [0.0998, 0.0, 0.0]
    assert manifest["metadata"]["raw_stereo_params"]["baseline_m"] == -0.0998
    assert manifest["camera_model"]["is_rectified"] is True
    assert manifest["timestamps"]["left_path"].endswith("timestamps/left_timestamps.txt")
