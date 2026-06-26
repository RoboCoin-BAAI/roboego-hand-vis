import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.stereo_video import (  # noqa: E402
    SegmentQualityConfig,
    SummaryQualityConfig,
    run_segment_quality_check,
    run_summary_quality_check,
)


def write_tiny_video(path: Path, width: int = 32, height: int = 24, frames: int = 4, fps: float = 4.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for idx in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
        frame[:, :, 1] = idx * 20
        frame[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
        writer.write(frame)
    writer.release()


def write_manifest(path: Path, left: Path, right: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_name": "tiny",
                "sequence_name": "test",
                "videos": {
                    "left": {"path": str(left)},
                    "right": {"path": str(right)},
                },
            }
        ),
        encoding="utf-8",
    )


def test_summary_quality_check_reads_manifest_and_writes_report(tmp_path):
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    manifest = tmp_path / "dataset_manifest.json"
    output_dir = tmp_path / "quality_summary"
    write_tiny_video(left)
    write_tiny_video(right)
    write_manifest(manifest, left, right)

    report = run_summary_quality_check(
        manifest_path=manifest,
        output_dir=output_dir,
        config=SummaryQualityConfig(sample_count=3, min_orb_matches=0),
    )

    assert report["left_video"]["path"] == str(left)
    assert report["right_video"]["path"] == str(right)
    assert report["summary"]["sample_count"] == 3
    assert (output_dir / "stereo_quality_report.json").exists()
    assert (output_dir / "stereo_quality_samples.csv").exists()
    assert (output_dir / "contact_sheet.png").exists()


def test_segment_quality_check_reads_manifest_and_writes_bad_segments(tmp_path):
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    manifest = tmp_path / "dataset_manifest.json"
    output_dir = tmp_path / "quality_segments"
    write_tiny_video(left, frames=5, fps=5.0)
    write_tiny_video(right, frames=5, fps=5.0)
    write_manifest(manifest, left, right)

    report = run_segment_quality_check(
        manifest_path=manifest,
        output_dir=output_dir,
        config=SegmentQualityConfig(sample_every_s=0.2, orb_every_s=0.4, min_orb_matches=0),
    )

    assert report["sampled_frame_count"] >= 4
    assert "bad_segments" in report
    assert (output_dir / "bad_segments.json").exists()
    assert (output_dir / "bad_frames.csv").exists()
    assert (output_dir / "all_sampled_frames.csv").exists()
    assert (output_dir / "contact_sheet_bad.png").exists()
