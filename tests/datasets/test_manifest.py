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
