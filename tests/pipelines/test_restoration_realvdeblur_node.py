from pathlib import Path

from roboego_hand_vis.pipelines.restoration.nodes import run_realvdeblur_stereo_node


def test_realvdeblur_stereo_node_runs_wrapper_script(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.restoration.nodes.subprocess.run", fake_run)
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"
    output_dir = tmp_path / "handwin/deblur_realvdeblur"

    result = run_realvdeblur_stereo_node(
        {
            "enabled": True,
            "python": "python",
            "script": "scripts/restoration/run_realvdeblur_stereo_manifest.py",
            "manifest": str(manifest),
            "output_dir": str(output_dir),
            "realvdeblur_python": "/envs/realvdeblur/bin/python",
            "realvdeblur_root": "/models/RealVDeblur",
            "wan_model_dir": "/models/RealVDeblur/checkpoints/Wan2.1-T2V-1.3B",
            "checkpoint": "/models/RealVDeblur/checkpoints/realvdeblur_dmd.safetensors",
            "height": 1072,
            "width": 1920,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "num_inference_steps": 1,
            "enable_twm": True,
            "temporal_window_size": 21,
            "overwrite": True,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "python",
                "scripts/restoration/run_realvdeblur_stereo_manifest.py",
                "--manifest",
                str(manifest),
                "--output_dir",
                str(output_dir),
                "--realvdeblur_python",
                "/envs/realvdeblur/bin/python",
                "--realvdeblur_root",
                "/models/RealVDeblur",
                "--wan_model_dir",
                "/models/RealVDeblur/checkpoints/Wan2.1-T2V-1.3B",
                "--checkpoint",
                "/models/RealVDeblur/checkpoints/realvdeblur_dmd.safetensors",
                "--height",
                "1072",
                "--width",
                "1920",
                "--device",
                "cuda:0",
                "--dtype",
                "bfloat16",
                "--num_inference_steps",
                "1",
                "--enable_twm",
                "--temporal_window_size",
                "21",
                "--overwrite",
            ],
        }
    ]
    assert result == {
        "stage": "realvdeblur_stereo",
        "status": "complete",
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "deblur_manifest": str(output_dir / "dataset_manifest" / "dataset_manifest.json"),
    }


def test_realvdeblur_stereo_node_defaults_output_next_to_manifest(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("roboego_hand_vis.pipelines.restoration.nodes.subprocess.run", lambda args, check: calls.append(args))
    manifest = tmp_path / "handwin/dataset_manifest/dataset_manifest.json"

    result = run_realvdeblur_stereo_node({"enabled": True, "manifest": str(manifest)})

    assert result["output_dir"] == str(tmp_path / "handwin/deblur_realvdeblur")
    assert calls[0][calls[0].index("--output_dir") + 1] == str(tmp_path / "handwin/deblur_realvdeblur")


def test_realvdeblur_stereo_node_skips_when_disabled():
    assert run_realvdeblur_stereo_node({"enabled": False}) == {
        "stage": "realvdeblur_stereo",
        "status": "skipped",
        "reason": "disabled",
    }
