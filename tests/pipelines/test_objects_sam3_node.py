from roboego_hand_vis.pipelines.objects.nodes import run_sam3_hand_object_segmentation_node


def test_sam3_hand_object_segmentation_node_skips_when_disabled():
    assert run_sam3_hand_object_segmentation_node({"enabled": False}) == {
        "stage": "sam3_hand_object_segmentation",
        "status": "skipped",
        "reason": "disabled",
    }


def test_sam3_hand_object_segmentation_node_requires_window_dir_when_enabled():
    try:
        run_sam3_hand_object_segmentation_node({"enabled": True})
    except ValueError as exc:
        assert "hand_window_dir" in str(exc)
    else:
        raise AssertionError("expected missing hand_window_dir to fail")


def test_sam3_hand_object_segmentation_node_runs_existing_script(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.objects.nodes.subprocess.run", fake_run)

    hand_window_dir = tmp_path / "segments/seg_0000/hand_windows/handwin_0000/hamer_windows_left_v1/run"
    stage1_actions = tmp_path / "annotation/qwen_frame_annotations_left_v4.checkpoint.json"
    windows_json = tmp_path / "annotation/hand_estimation_windows_left_v1.json"
    result = run_sam3_hand_object_segmentation_node(
        {
            "enabled": True,
            "python": "/env/ego/bin/python",
            "hand_window_dir": str(hand_window_dir),
            "stage1_actions": str(stage1_actions),
            "windows_json": str(windows_json),
            "sam3_python": "/env/py312/bin/python",
            "checkpoint": "/models/sam3.pt",
            "fps": 30,
            "auto_from_stage1": True,
            "dry_run": True,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/env/ego/bin/python",
                "scripts/object/run_sam3_hand_clip_objects.py",
                "--hand_window_dir",
                str(hand_window_dir),
                "--stage1_actions",
                str(stage1_actions),
                "--windows_json",
                str(windows_json),
                "--python",
                "/env/py312/bin/python",
                "--fps",
                "30",
                "--checkpoint",
                "/models/sam3.pt",
                "--auto_from_stage1",
                "--dry_run",
            ],
        }
    ]
    assert result["status"] == "complete"
    assert result["processed_count"] == 1
    assert result["hand_window_dirs"] == [str(hand_window_dir)]
