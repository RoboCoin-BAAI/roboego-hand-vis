import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.tracking_interface import (  # noqa: E402
    build_track_report_payload,
    load_tracking_inputs_from_manifest,
    status_layer,
)


def write_manifest(path: Path, left_video: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_name": "wuxi",
                "sequence_name": "default",
                "reference_camera": "left",
                "videos": {
                    "left": {"path": str(left_video), "fps": 60.0, "width": 1624, "height": 1232},
                    "right": {"path": str(path.parent / "right.mp4")},
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_tracking_inputs_from_manifest_uses_left_video(tmp_path):
    left_video = tmp_path / "stereo_left.mp4"
    manifest = tmp_path / "dataset_manifest.json"
    write_manifest(manifest, left_video)

    inputs = load_tracking_inputs_from_manifest(manifest)

    assert inputs.video_path == left_video.resolve()
    assert inputs.dataset_name == "wuxi"
    assert inputs.video_info["fps"] == 60.0


def test_status_layer_names_normal_recovered_and_smooth():
    assert status_layer({"status": "detected"}) == "normal"
    assert status_layer({"status": "recovered"}) == "recovered"
    assert status_layer({"status": "interpolated"}) == "smooth"
    assert status_layer(None) == "missing"


def test_build_track_report_payload_records_standard_layers(tmp_path):
    left_video = tmp_path / "stereo_left.mp4"
    manifest = tmp_path / "dataset_manifest.json"
    write_manifest(manifest, left_video)
    inputs = load_tracking_inputs_from_manifest(manifest)
    tracks = {
        "ego_left": [
            {"status": "detected", "frame_idx": 0},
            {"status": "interpolated", "frame_idx": 1},
        ],
        "ego_right": [
            {"status": "recovered", "frame_idx": 0},
            None,
        ],
    }

    payload = build_track_report_payload(
        inputs=inputs,
        alignment_report=tmp_path / "alignment.json",
        depth_dir=tmp_path / "depth",
        output_dir=tmp_path / "track",
        parameters={"start": 0, "num_frames": 2},
        summary={"ego_left": {"total": 2}},
        tracks=tracks,
    )

    assert payload["schema_version"] == "1.0"
    assert payload["manifest_path"] == str(manifest.resolve())
    assert payload["status_layers"]["detected"] == "normal"
    assert payload["tracks"]["ego_left"][0]["layer"] == "normal"
    assert payload["tracks"]["ego_left"][1]["layer"] == "smooth"
    assert payload["tracks"]["ego_right"][0]["layer"] == "recovered"
    assert payload["tracks"]["ego_right"][1] is None
