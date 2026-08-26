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
            "write_avi": True,
            "write_preview": True,
            "preview_codec": "mp4v",
            "dry_run_avi": True,
            "delete_npy": False,
            "delete_png": True,
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
                "--write_avi",
                "--write_preview",
                "--preview_codec",
                "mp4v",
                "--dry_run_avi",
                "--delete_png",
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
