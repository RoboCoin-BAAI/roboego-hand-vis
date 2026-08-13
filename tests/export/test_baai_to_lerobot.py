import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export.baai_to_lerobot import (
    build_camera_parameters_annotation,
    parse_quality_report_specs,
    write_quality_annotations,
)


def test_build_camera_parameters_annotation_preserves_manifest_camera_fields():
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "generic_stereo",
        "sequence_name": "baai_ego_task_001",
        "reference_camera": "left",
        "videos": {
            "left": {"path": "/data/left.mp4", "width": 1600, "height": 900, "fps": 30.0},
            "right": {"path": "/data/right.mp4", "width": 1600, "height": 900, "fps": 30.0},
        },
        "camera_model": {
            "type": "pinhole",
            "is_rectified": True,
            "is_undistorted": True,
            "calibration_source": "/data/camera.json",
        },
        "intrinsics": {
            "left": {"fx": 474.0, "fy": 474.0, "cx": 797.0, "cy": 450.0},
            "right": {"fx": 474.0, "fy": 474.0, "cx": 797.0, "cy": 450.0},
        },
        "extrinsics": {
            "left_to_right": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0.063, 0, 0]},
            "baseline_m": 0.063,
        },
        "timestamps": {"type": "file", "left_path": "/data/left.txt", "right_path": "/data/right.txt"},
    }

    annotation = build_camera_parameters_annotation(manifest, Path("/tmp/manifest.json"))

    assert annotation["schema_version"] == "1.0"
    assert annotation["source_manifest"] == "/tmp/manifest.json"
    assert annotation["reference_camera"] == "left"
    assert annotation["camera_model"]["is_rectified"] is True
    assert annotation["intrinsics"]["left"]["fx"] == 474.0
    assert annotation["extrinsics"]["baseline_m"] == 0.063
    assert annotation["timestamps"]["left_path"] == "/data/left.txt"
    json.dumps(annotation)


def test_write_quality_annotations_copies_named_reports_to_annotation_root(tmp_path):
    report = tmp_path / "stereo_quality_report.json"
    report.write_text(json.dumps({"summary": {"passed": True}}), encoding="utf-8")
    output_root = tmp_path / "lerobot"

    written = write_quality_annotations([("stereo", report)], output_root)

    copied = output_root / "annotation" / "stereo_quality_report.json"
    index_path = output_root / "annotation" / "quality_reports.json"

    assert written == [copied]
    assert json.loads(copied.read_text(encoding="utf-8"))["summary"]["passed"] is True
    assert not index_path.exists()


def test_parse_quality_report_specs_accepts_name_equals_path(tmp_path):
    report = tmp_path / "depth_quality_report.json"
    report.write_text("{}", encoding="utf-8")

    specs = parse_quality_report_specs([f"depth={report}"], tmp_path)

    assert specs == [("depth", report.resolve())]
