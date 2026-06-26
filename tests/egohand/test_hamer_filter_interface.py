import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.hamer_filter import (  # noqa: E402
    build_report_metadata,
    depth_quality_warnings_for_frame,
    load_depth_quality_warnings,
    load_hamer_filter_inputs_from_manifest,
)


def write_manifest(path: Path, left_video: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_name": "wuxi",
                "sequence_name": "default",
                "reference_camera": "left",
                "videos": {
                    "left": {
                        "path": str(left_video),
                        "width": 1624,
                        "height": 1232,
                        "fps": 60.0,
                        "frame_count": 2830,
                    },
                    "right": {"path": str(path.parent / "right.mp4")},
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_hamer_filter_inputs_from_manifest_uses_left_video(tmp_path):
    left_video = tmp_path / "stereo_left.mp4"
    manifest = tmp_path / "dataset_manifest.json"
    write_manifest(manifest, left_video)

    inputs = load_hamer_filter_inputs_from_manifest(manifest)

    assert inputs.video_path == left_video.resolve()
    assert inputs.dataset_name == "wuxi"
    assert inputs.sequence_name == "default"
    assert inputs.video_info["fps"] == 60.0


def test_depth_quality_warnings_for_frame_reads_frame_warnings(tmp_path):
    report = tmp_path / "depth_quality_report.json"
    report.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame": 10, "warnings": ["stereo_quality_p90_abs_dy"]},
                    {"frame": 11, "warnings": []},
                ]
            }
        ),
        encoding="utf-8",
    )

    warnings = load_depth_quality_warnings(report)

    assert depth_quality_warnings_for_frame(10, warnings) == ["stereo_quality_p90_abs_dy"]
    assert depth_quality_warnings_for_frame(11, warnings) == []
    assert depth_quality_warnings_for_frame(12, warnings) == []


def test_build_report_metadata_records_manifest_depth_and_hamer_config(tmp_path):
    left_video = tmp_path / "stereo_left.mp4"
    manifest = tmp_path / "dataset_manifest.json"
    write_manifest(manifest, left_video)
    inputs = load_hamer_filter_inputs_from_manifest(manifest)

    meta = build_report_metadata(
        inputs=inputs,
        depth_dir=tmp_path / "depth",
        depth_quality_report=tmp_path / "depth_quality_report.json",
        output_dir=tmp_path / "out",
        frame_range={"start": 0, "num_frames": 2, "stride": 1},
        filter_config={"min_area_ratio": 0.002},
        hamer_config={"body_detector": "regnety", "batch_size": 1},
    )

    assert meta["manifest_path"] == str(manifest.resolve())
    assert meta["dataset_name"] == "wuxi"
    assert meta["video"] == str(left_video.resolve())
    assert meta["depth_dir"] == str((tmp_path / "depth").resolve())
    assert meta["hamer_config"]["body_detector"] == "regnety"
