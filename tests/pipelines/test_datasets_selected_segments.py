from pathlib import Path

from roboego_hand_vis.pipelines.datasets.nodes import (
    run_export_hand_window_segments_node,
    run_export_selected_video_segments_node,
    run_prepare_selected_segments_node,
    run_resolve_dataset_manifest_node,
)


def test_export_selected_video_segments_node_skips_when_disabled():
    result = run_export_selected_video_segments_node({"enabled": False})

    assert result == {
        "stage": "export_selected_video_segments",
        "status": "skipped",
        "reason": "disabled",
    }


def test_resolve_dataset_manifest_node_writes_manifest(monkeypatch, tmp_path):
    calls = []

    def fake_resolve_dataset_manifest(config):
        calls.append(
            {
                "dataset": config.dataset,
                "root": config.root,
                "output_dir": config.output_dir,
                "sequence_name": config.sequence_name,
                "left_video": config.left_video,
                "assume_rectified": config.assume_rectified,
            }
        )
        return {"dataset_name": config.dataset, "sequence_name": config.sequence_name}

    def fake_write_manifest(manifest, output_dir):
        return output_dir / "dataset_manifest.json", output_dir / "dataset_summary.txt"

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.datasets.nodes.resolve_dataset_manifest",
        fake_resolve_dataset_manifest,
    )
    monkeypatch.setattr("roboego_hand_vis.pipelines.datasets.nodes.write_manifest", fake_write_manifest)

    result = run_resolve_dataset_manifest_node(
        {
            "enabled": True,
            "dataset": "ropedia",
            "root": str(tmp_path / "ropedia_data"),
            "output_dir": str(tmp_path / "tmp/ropedia_dataset_manifest"),
            "sequence_name": "ropedia_test",
            "left_video": str(tmp_path / "left.mp4"),
            "assume_rectified": False,
        }
    )

    assert calls == [
        {
            "dataset": "ropedia",
            "root": tmp_path / "ropedia_data",
            "output_dir": tmp_path / "tmp/ropedia_dataset_manifest",
            "sequence_name": "ropedia_test",
            "left_video": tmp_path / "left.mp4",
            "assume_rectified": False,
        }
    ]
    assert result == {
        "stage": "resolve_dataset_manifest",
        "status": "complete",
        "dataset": "ropedia",
        "root": str(tmp_path / "ropedia_data"),
        "manifest_path": str(tmp_path / "tmp/ropedia_dataset_manifest/dataset_manifest.json"),
        "summary_path": str(tmp_path / "tmp/ropedia_dataset_manifest/dataset_summary.txt"),
    }


def test_resolve_dataset_manifest_node_skips_when_disabled():
    assert run_resolve_dataset_manifest_node({"enabled": False}) == {
        "stage": "resolve_dataset_manifest",
        "status": "skipped",
        "reason": "disabled",
    }


def test_export_selected_video_segments_node_calls_exporter(monkeypatch, tmp_path):
    calls = []

    def fake_export_selected_video_segments(**kwargs):
        calls.append(kwargs)
        return {
            "source_session": str(kwargs["session_dir"]),
            "output_dir": str(kwargs["output_dir"]),
            "segment_count": 2,
            "segments": [{"segment_id": "seg_0000"}, {"segment_id": "seg_0001"}],
        }

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.datasets.nodes.export_selected_video_segments",
        fake_export_selected_video_segments,
    )

    result = run_export_selected_video_segments_node(
        {
            "enabled": True,
            "session": str(tmp_path / "session"),
            "processing_plan": str(tmp_path / "session/annotation/processing_plan_left_v1.json"),
            "output_dir": str(tmp_path / "session/segments"),
            "camera_streams": ["left", "right"],
            "fps": 30,
            "codec": "h264",
            "overwrite": True,
        }
    )

    assert calls == [
        {
            "session_dir": tmp_path / "session",
            "processing_plan_json": tmp_path / "session/annotation/processing_plan_left_v1.json",
            "output_dir": tmp_path / "session/segments",
            "camera_streams": ("left", "right"),
            "fps": 30,
            "codec": "h264",
            "overwrite": True,
        }
    ]
    assert result["status"] == "complete"
    assert result["segment_count"] == 2


def test_prepare_selected_segments_node_runs_existing_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.datasets.nodes.subprocess.run", fake_run)
    export_report = {
        "status": "complete",
        "output_dir": str(tmp_path / "session/segments"),
        "segment_count": 2,
    }

    result = run_prepare_selected_segments_node(
        export_report,
        {
            "enabled": True,
            "python": "python",
            "script": "scripts/datasets/prepare_selected_segments.py",
            "session": str(tmp_path / "session"),
            "segments_root": str(tmp_path / "session/segments"),
            "roboegopipe_root": "/tmp/RoboEgoPipe",
            "model": "equidistant",
            "balance": 0.0,
            "overwrite": True,
        },
    )

    assert calls[0]["check"] is True
    assert calls[0]["args"] == [
        "python",
        "scripts/datasets/prepare_selected_segments.py",
        "--session",
        str(tmp_path / "session"),
        "--segments_root",
        str(tmp_path / "session/segments"),
        "--roboegopipe_root",
        "/tmp/RoboEgoPipe",
        "--model",
        "equidistant",
        "--balance",
        "0.0",
        "--overwrite",
    ]
    assert result == {
        "stage": "prepare_selected_segments",
        "status": "complete",
        "session": str(tmp_path / "session"),
        "segments_root": str(tmp_path / "session/segments"),
    }


def test_prepare_selected_segments_node_skips_when_export_not_complete():
    result = run_prepare_selected_segments_node({"status": "skipped"}, {"enabled": True})

    assert result == {
        "stage": "prepare_selected_segments",
        "status": "skipped",
        "reason": "export_selected_video_segments_not_complete",
    }


def test_export_hand_window_segments_node_runs_after_prepare(monkeypatch, tmp_path):
    calls = []

    def fake_export_hand_window_segments(**kwargs):
        calls.append(kwargs)
        return {"window_count": 2, "windows": [{"window_id": "handwin_0000"}, {"window_id": "handwin_0001"}]}

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.datasets.nodes.export_hand_window_segments",
        fake_export_hand_window_segments,
    )

    result = run_export_hand_window_segments_node(
        {"status": "complete", "segments_root": str(tmp_path / "session/segments")},
        {
            "enabled": True,
            "segments_root": str(tmp_path / "session/segments"),
            "hand_windows_json": str(tmp_path / "session/annotation/hand_estimation_windows_left_v1.json"),
            "overwrite": True,
            "codec": "mp4v",
        },
    )

    assert calls == [
        {
            "segments_root": tmp_path / "session/segments",
            "hand_windows_json": tmp_path / "session/annotation/hand_estimation_windows_left_v1.json",
            "overwrite": True,
            "codec": "mp4v",
        }
    ]
    assert result["stage"] == "export_hand_window_segments"
    assert result["status"] == "complete"
    assert result["window_count"] == 2


def test_export_hand_window_segments_node_skips_when_prepare_not_complete():
    result = run_export_hand_window_segments_node({"status": "skipped"}, {"enabled": True})

    assert result == {
        "stage": "export_hand_window_segments",
        "status": "skipped",
        "reason": "prepare_selected_segments_not_complete",
    }
