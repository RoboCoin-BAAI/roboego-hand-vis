import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.depth.foundation import load_foundation_inputs_from_manifest  # noqa: E402
from roboego_hand_vis.depth.quality import DepthQualityConfig, run_depth_quality_check  # noqa: E402


def write_video(path: Path, width: int = 16, height: int = 12, frames: int = 3, fps: float = 3.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for idx in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :] * 4
        frame[:, :, 1] = idx * 40
        frame[:, :, 2] = 120
        writer.write(frame)
    writer.release()


def write_manifest(path: Path, left: Path, right: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_name": "tiny",
                "sequence_name": "depth_test",
                "videos": {
                    "left": {"path": str(left), "width": 16, "height": 12, "fps": 3.0, "frame_count": 3},
                    "right": {"path": str(right), "width": 16, "height": 12, "fps": 3.0, "frame_count": 3},
                },
                "intrinsics": {
                    "left": {"fx": 20.0, "fy": 21.0, "cx": 8.0, "cy": 6.0, "width": 16, "height": 12},
                    "right": {"fx": 20.0, "fy": 21.0, "cx": 8.0, "cy": 6.0, "width": 16, "height": 12},
                },
                "extrinsics": {"baseline_m": 0.06},
            }
        ),
        encoding="utf-8",
    )


def test_load_foundation_inputs_from_manifest_reads_paths_intrinsics_and_baseline(tmp_path):
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    manifest = tmp_path / "dataset_manifest.json"
    write_video(left)
    write_video(right)
    write_manifest(manifest, left, right)

    inputs = load_foundation_inputs_from_manifest(manifest)

    assert inputs.left_video == left.resolve()
    assert inputs.right_video == right.resolve()
    assert inputs.intrinsics["fx"] == 20.0
    assert inputs.intrinsics["fy"] == 21.0
    assert inputs.baseline_m == 0.06


def test_depth_quality_check_outputs_report_overlay_and_pointcloud_preview(tmp_path):
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    manifest = tmp_path / "dataset_manifest.json"
    depth_dir = tmp_path / "depth"
    output_dir = tmp_path / "depth_quality"
    depth_dir.mkdir()
    write_video(left)
    write_video(right)
    write_manifest(manifest, left, right)
    for frame_idx in range(2):
        depth = np.full((12, 16), 0.8 + frame_idx * 0.1, dtype=np.float32)
        depth[:, :2] = np.nan
        np.save(depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy", depth)

    report = run_depth_quality_check(
        manifest_path=manifest,
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthQualityConfig(start=0, num_frames=2, point_stride=4, max_preview_points=20),
    )

    assert report["summary"]["frame_count"] == 2
    assert report["summary"]["valid_depth_ratio_avg"] > 0.8
    assert report["frames"][0]["depth_median_m"] == 0.800000011920929
    assert (output_dir / "depth_quality_report.json").exists()
    assert (output_dir / "vis" / "frame_000000_depth_overlay.jpg").exists()
    assert (output_dir / "pointcloud_preview" / "frame_000000_pointcloud_preview.ply").exists()


def test_depth_quality_check_marks_stereo_quality_segment_warnings(tmp_path):
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    manifest = tmp_path / "dataset_manifest.json"
    depth_dir = tmp_path / "depth"
    output_dir = tmp_path / "depth_quality"
    stereo_report = tmp_path / "bad_segments.json"
    depth_dir.mkdir()
    write_video(left)
    write_video(right)
    write_manifest(manifest, left, right)
    np.save(depth_dir / "frame_000000_foundation_depth_m.npy", np.full((12, 16), 0.8, dtype=np.float32))
    stereo_report.write_text(
        json.dumps(
            {
                "bad_segments": [
                    {"start_frame": 0, "end_frame": 0, "reasons": ["sharpness_low", "p90_abs_dy"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_depth_quality_check(
        manifest_path=manifest,
        depth_dir=depth_dir,
        output_dir=output_dir,
        stereo_quality_report=stereo_report,
        config=DepthQualityConfig(start=0, num_frames=1, point_stride=4, max_preview_points=20),
    )

    assert "stereo_quality_sharpness_low" in report["frames"][0]["warnings"]
    assert "stereo_quality_p90_abs_dy" in report["frames"][0]["warnings"]
