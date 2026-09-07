import json
from pathlib import Path

from roboego_hand_vis.pipelines.depth.nodes import (
    run_archive_depth_cache_node,
    run_fast_foundation_stereo_depth_node,
    run_foundation_stereo_depth_node,
)


def test_foundation_stereo_depth_node_runs_existing_script(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.depth.nodes.subprocess.run", fake_run)
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"
    output_dir = tmp_path / "handwin/foundation_depth"

    result = run_foundation_stereo_depth_node(
        {
            "enabled": True,
            "python": "python",
            "script": "scripts/run_foundation_stereo_video.py",
            "manifest": str(manifest),
            "output_dir": str(output_dir),
            "foundation_root": "/models/FoundationStereo",
            "ckpt": "/models/FoundationStereo/model.pth",
            "start": 2,
            "num_frames": 9,
            "scale": 0.5,
            "valid_iters": 16,
            "max_depth_m": 6.0,
            "save_vis": False,
            "overwrite": True,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "python",
                "scripts/run_foundation_stereo_video.py",
                "--manifest",
                str(manifest),
                "--output_dir",
                str(output_dir),
                "--foundation_root",
                "/models/FoundationStereo",
                "--ckpt",
                "/models/FoundationStereo/model.pth",
                "--start",
                "2",
                "--num_frames",
                "9",
                "--scale",
                "0.5",
                "--valid_iters",
                "16",
                "--max_depth_m",
                "6.0",
                "--no-save_vis",
                "--overwrite",
            ],
        }
    ]
    assert result == {
        "stage": "foundation_stereo_depth",
        "status": "complete",
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "meta_path": str(output_dir / "foundation_stereo_video_meta.json"),
    }


def test_foundation_stereo_depth_node_defaults_output_next_to_manifest(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("roboego_hand_vis.pipelines.depth.nodes.subprocess.run", lambda args, check: calls.append(args))
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"

    result = run_foundation_stereo_depth_node({"enabled": True, "manifest": str(manifest), "num_frames": 1})

    assert result["output_dir"] == str(tmp_path / "handwin/depth/foundation")
    assert calls[0][calls[0].index("--output_dir") + 1] == str(tmp_path / "handwin/depth/foundation")


def test_foundation_stereo_depth_node_skips_when_disabled():
    assert run_foundation_stereo_depth_node({"enabled": False}) == {
        "stage": "foundation_stereo_depth",
        "status": "skipped",
        "reason": "disabled",
    }


def test_fast_foundation_stereo_depth_node_runs_fast_script(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.depth.nodes.subprocess.run", fake_run)
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"
    output_dir = tmp_path / "handwin/fast_foundation_depth"

    result = run_fast_foundation_stereo_depth_node(
        {
            "enabled": True,
            "python": "/envs/py312/bin/python",
            "script": "scripts/run_fast_foundation_stereo_video.py",
            "manifest": str(manifest),
            "output_dir": str(output_dir),
            "fast_foundation_root": "/models/Fast-FoundationStereo",
            "ckpt": "/models/Fast-FoundationStereo/model.pth",
            "start": 91,
            "num_frames": 1,
            "scale": 0.5,
            "valid_iters": 4,
            "max_disp": 192,
            "max_depth_m": 4.0,
            "save_depth_png": True,
            "save_npy": False,
            "save_vis": True,
            "overwrite": True,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/envs/py312/bin/python",
                "scripts/run_fast_foundation_stereo_video.py",
                "--manifest",
                str(manifest),
                "--output_dir",
                str(output_dir),
                "--fast_foundation_root",
                "/models/Fast-FoundationStereo",
                "--ckpt",
                "/models/Fast-FoundationStereo/model.pth",
                "--start",
                "91",
                "--num_frames",
                "1",
                "--scale",
                "0.5",
                "--valid_iters",
                "4",
                "--max_disp",
                "192",
                "--max_depth_m",
                "4.0",
                "--save_depth_png",
                "--no-save_npy",
                "--save_vis",
                "--overwrite",
            ],
        }
    ]
    assert result == {
        "stage": "fast_foundation_stereo_depth",
        "status": "complete",
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "meta_path": str(output_dir / "fast_foundation_stereo_video_meta.json"),
    }


def test_fast_foundation_stereo_depth_node_allows_usable_stereo_warning(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("roboego_hand_vis.pipelines.depth.nodes.subprocess.run", lambda args, check: calls.append(args))
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"
    stereo_report = tmp_path / "handwin/quality/stereo_summary/stereo_quality_report.json"
    stereo_report.parent.mkdir(parents=True)
    stereo_report.write_text(
        json.dumps(
            {
                "summary": {
                    "passed": False,
                    "usable_for_depth": True,
                    "depth_risk_level": "high_risk",
                    "depth_risk_reasons": ["geometry_warning"],
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_fast_foundation_stereo_depth_node(
        {
            "enabled": True,
            "manifest": str(manifest),
            "stereo_quality_report": str(stereo_report),
            "num_frames": 1,
        }
    )

    assert calls
    assert result["stereo_quality_report"] == str(stereo_report)
    assert result["stereo_quality_passed"] is False
    assert result["usable_for_depth"] is True
    assert result["depth_risk_level"] == "high_risk"
    assert result["depth_risk_reasons"] == ["geometry_warning"]


def test_fast_foundation_stereo_depth_node_blocks_unusable_stereo_report(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("roboego_hand_vis.pipelines.depth.nodes.subprocess.run", lambda args, check: calls.append(args))
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"
    stereo_report = tmp_path / "handwin/quality/stereo_summary/stereo_quality_report.json"
    stereo_report.parent.mkdir(parents=True)
    stereo_report.write_text(
        json.dumps({"summary": {"passed": False, "usable_for_depth": False, "depth_risk_level": "blocked"}}),
        encoding="utf-8",
    )

    result = run_fast_foundation_stereo_depth_node(
        {
            "enabled": True,
            "manifest": str(manifest),
            "stereo_quality_report": str(stereo_report),
            "num_frames": 1,
        }
    )

    assert calls == []
    assert result["status"] == "blocked"
    assert result["reason"] == "stereo_quality_not_usable_for_depth"
    assert result["stereo_quality_report"] == str(stereo_report)


def test_fast_foundation_stereo_depth_node_defaults_output_next_to_manifest(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("roboego_hand_vis.pipelines.depth.nodes.subprocess.run", lambda args, check: calls.append(args))
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"

    result = run_fast_foundation_stereo_depth_node({"enabled": True, "manifest": str(manifest), "num_frames": 1})

    assert result["output_dir"] == str(tmp_path / "handwin/depth/fast_foundation")
    assert calls[0][calls[0].index("--output_dir") + 1] == str(tmp_path / "handwin/depth/fast_foundation")


def test_fast_foundation_stereo_depth_node_skips_when_disabled():
    assert run_fast_foundation_stereo_depth_node({"enabled": False}) == {
        "stage": "fast_foundation_stereo_depth",
        "status": "skipped",
        "reason": "disabled",
    }


def test_archive_depth_cache_node_runs_existing_script(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.depth.nodes.subprocess.run", fake_run)
    depth_dir = tmp_path / "handwin/depth/fast_foundation"
    output_dir = tmp_path / "handwin/depth/fast_foundation_archive"

    result = run_archive_depth_cache_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "script": "scripts/depth/archive_depth_cache.py",
            "depth_dir": str(depth_dir),
            "output_dir": str(output_dir),
            "max_depth_m": 2.5,
            "invalid_value": 0,
            "fps": 30,
            "write_png": True,
            "write_avi": True,
            "write_preview": True,
            "preview_mode": "near",
            "preview_near_min_m": 0.15,
            "preview_near_max_m": 3.0,
            "preview_contrast_percentile": 2.0,
            "preview_codec": "mp4v",
            "ffv1_container": "mkv",
            "dry_run_avi": True,
            "delete_npy": False,
            "delete_png": True,
            "write_hevc12": True,
            "hevc12_crf": 0,
            "hevc12_preset": "slow",
            "hevc12_lossless": True,
            "write_valid_mask": True,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/envs/ego/bin/python",
                "scripts/depth/archive_depth_cache.py",
                "--depth_dir",
                str(depth_dir),
                "--output_dir",
                str(output_dir),
                "--max_depth_m",
                "2.5",
                "--invalid_value",
                "0",
                "--fps",
                "30",
                "--write_png",
                "--write_avi",
                "--write_preview",
                "--preview_mode",
                "near",
                "--preview_near_min_m",
                "0.15",
                "--preview_near_max_m",
                "3.0",
                "--preview_contrast_percentile",
                "2.0",
                "--preview_codec",
                "mp4v",
                "--ffv1_container",
                "mkv",
                "--dry_run_avi",
                "--delete_png",
                "--write_hevc12",
                "--hevc12_crf",
                "0",
                "--hevc12_preset",
                "slow",
                "--hevc12_lossless",
                "--write_valid_mask",
            ],
        }
    ]
    assert result == {
        "stage": "archive_depth_cache",
        "status": "complete",
        "depth_dir": str(depth_dir),
        "output_dir": str(output_dir),
        "metadata_json": str(output_dir / "depth_archive_meta.json"),
    }


def test_archive_depth_cache_node_skips_when_disabled():
    assert run_archive_depth_cache_node({"enabled": False}) == {
        "stage": "archive_depth_cache",
        "status": "skipped",
        "reason": "disabled",
    }
