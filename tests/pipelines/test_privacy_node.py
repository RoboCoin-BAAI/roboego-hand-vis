from __future__ import annotations

import json
from pathlib import Path

from roboego_hand_vis.pipelines.privacy.nodes import run_egoblur_temporal_face_blur_node


def test_egoblur_temporal_face_blur_node_runs_configured_python_script(monkeypatch, tmp_path: Path):
    output_json = tmp_path / "report.json"
    calls = []

    def fake_run(args, check):
        calls.append(args)
        output_json.write_text(json.dumps({"stage": "egoblur_temporal_face_blur", "status": "complete"}), encoding="utf-8")

    monkeypatch.setattr("subprocess.run", fake_run)

    report = run_egoblur_temporal_face_blur_node(
        {
            "enabled": True,
            "python": "/envs/py312/bin/python",
            "script": "scripts/privacy/run_egoblur_temporal_face_blur.py",
            "input_video": str(tmp_path / "left.mp4"),
            "output_video": str(tmp_path / "left_blur.mp4"),
            "output_json": str(output_json),
            "egoblur_root": "/path/to/EgoBlur",
            "face_model_path": "/path/to/EgoBlur/models/ego_blur_face_gen2.jit",
            "detect_every_n_frames": 3,
            "face_model_score_threshold": 0.35,
            "max_gap_frames": 12,
            "smooth_window": 5,
        }
    )

    assert report["status"] == "complete"
    assert calls
    assert calls[0][:2] == ["/envs/py312/bin/python", "scripts/privacy/run_egoblur_temporal_face_blur.py"]
    assert "--detect_every_n_frames" in calls[0]
    assert "3" in calls[0]
