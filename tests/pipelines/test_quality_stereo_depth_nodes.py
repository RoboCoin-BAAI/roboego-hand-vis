from pathlib import Path

from roboego_hand_vis.pipelines.quality.nodes import (
    run_depth_numeric_quality_node,
    run_depth_quality_node,
    run_stereo_segment_quality_node,
    run_stereo_summary_quality_node,
)
from roboego_hand_vis.pipelines.quality.pipeline import create_pipeline


def test_depth_quality_pipeline_node_can_run_from_params_only():
    pipeline = create_pipeline().only_nodes("depth_quality")

    assert pipeline.inputs() == {"params:depth_quality"}


def test_depth_numeric_quality_pipeline_node_can_run_from_params_only():
    pipeline = create_pipeline().only_nodes("depth_numeric_quality")

    assert pipeline.inputs() == {"params:depth_numeric_quality"}


def test_stereo_summary_quality_node_defaults_output_next_to_manifest(monkeypatch, tmp_path):
    calls = []

    def fake_run_summary_quality_check(*, manifest_path, output_dir, config):
        calls.append({"manifest_path": manifest_path, "output_dir": output_dir, "sample_count": config.sample_count})
        return {"summary": {"passed": True}, "outputs": {"csv": str(output_dir / "stereo_quality_samples.csv")}}

    monkeypatch.setattr("roboego_hand_vis.pipelines.quality.nodes.run_summary_quality_check", fake_run_summary_quality_check)
    manifest = tmp_path / "seg/rectified/dataset_manifest/dataset_manifest.json"

    result = run_stereo_summary_quality_node({"enabled": True, "manifest": str(manifest), "sample_count": 7})

    assert calls == [
        {
            "manifest_path": manifest,
            "output_dir": tmp_path / "seg/rectified/quality/stereo_summary",
            "sample_count": 7,
        }
    ]
    assert result["status"] == "complete"
    assert result["report_path"] == str(tmp_path / "seg/rectified/quality/stereo_summary/stereo_quality_report.json")


def test_stereo_segment_quality_node_calls_segment_checker(monkeypatch, tmp_path):
    calls = []

    def fake_run_segment_quality_check(*, manifest_path, output_dir, config):
        calls.append(
            {
                "manifest_path": manifest_path,
                "output_dir": output_dir,
                "sample_every_s": config.sample_every_s,
                "orb_every_s": config.orb_every_s,
            }
        )
        return {"passed": False, "bad_segment_count": 1, "outputs": {}}

    monkeypatch.setattr("roboego_hand_vis.pipelines.quality.nodes.run_segment_quality_check", fake_run_segment_quality_check)
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"

    result = run_stereo_segment_quality_node(
        {"enabled": True, "manifest": str(manifest), "sample_every_s": 0.5, "orb_every_s": 2.0}
    )

    assert calls == [
        {
            "manifest_path": manifest,
            "output_dir": tmp_path / "handwin/quality/stereo_segments",
            "sample_every_s": 0.5,
            "orb_every_s": 2.0,
        }
    ]
    assert result["status"] == "complete"
    assert result["passed"] is False
    assert result["report_path"] == str(tmp_path / "handwin/quality/stereo_segments/bad_segments.json")


def test_depth_quality_node_uses_stereo_segment_report_when_available(monkeypatch, tmp_path):
    calls = []

    def fake_run_depth_quality_check(*, manifest_path, depth_dir, output_dir, config, stereo_quality_report):
        calls.append(
            {
                "manifest_path": manifest_path,
                "depth_dir": depth_dir,
                "output_dir": output_dir,
                "start": config.start,
                "num_frames": config.num_frames,
                "stereo_quality_report": stereo_quality_report,
            }
        )
        return {"summary": {"warning_frame_count": 2}, "outputs": {}}

    monkeypatch.setattr("roboego_hand_vis.pipelines.quality.nodes.run_depth_quality_check", fake_run_depth_quality_check)
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"
    depth_dir = tmp_path / "handwin/depth"
    stereo_result = {"status": "complete", "report_path": str(tmp_path / "handwin/quality/stereo_segments/bad_segments.json")}

    result = run_depth_quality_node(
        stereo_result,
        {"enabled": True, "manifest": str(manifest), "depth_dir": str(depth_dir), "start": 3, "num_frames": 5},
    )

    assert calls == [
        {
            "manifest_path": manifest,
            "depth_dir": depth_dir,
            "output_dir": tmp_path / "handwin/quality/depth",
            "start": 3,
            "num_frames": 5,
            "stereo_quality_report": tmp_path / "handwin/quality/stereo_segments/bad_segments.json",
        }
    ]
    assert result["status"] == "complete"
    assert result["report_path"] == str(tmp_path / "handwin/quality/depth/depth_quality_report.json")


def test_depth_numeric_quality_node_calls_numeric_checker(monkeypatch, tmp_path):
    calls = []

    def fake_run_depth_numeric_quality_check(*, depth_dir, output_dir, config):
        calls.append(
            {
                "depth_dir": depth_dir,
                "output_dir": output_dir,
                "start": config.start,
                "num_frames": config.num_frames,
                "max_median_delta_m": config.max_median_delta_m,
            }
        )
        return {"summary": {"warning_frame_count": 1}, "outputs": {}}

    monkeypatch.setattr("roboego_hand_vis.pipelines.quality.nodes.run_depth_numeric_quality_check", fake_run_depth_numeric_quality_check)
    depth_dir = tmp_path / "handwin/depth/fast_foundation"

    result = run_depth_numeric_quality_node(
        {"enabled": True, "depth_dir": str(depth_dir), "start": 2, "num_frames": 8, "max_median_delta_m": 0.7}
    )

    assert calls == [
        {
            "depth_dir": depth_dir,
            "output_dir": tmp_path / "handwin/quality/depth_numeric",
            "start": 2,
            "num_frames": 8,
            "max_median_delta_m": 0.7,
        }
    ]
    assert result["status"] == "complete"
    assert result["report_path"] == str(tmp_path / "handwin/quality/depth_numeric/depth_numeric_quality_report.json")


def test_quality_nodes_skip_when_disabled():
    assert run_stereo_summary_quality_node({"enabled": False})["status"] == "skipped"
    assert run_stereo_segment_quality_node({"enabled": False})["status"] == "skipped"
    assert run_depth_quality_node({}, {"enabled": False})["status"] == "skipped"
    assert run_depth_numeric_quality_node({"enabled": False})["status"] == "skipped"
