import json

import cv2
import numpy as np
import pytest

from roboego_hand_vis.fast_foundation_depth.database_run import run_pending


class Service:
    def __init__(self, paths):
        self.rows = [dict(id=i, local_path=p) for i, p in enumerate(paths)]
        self.results = []

    def claim_fast_foundation_depth(self):
        return self.rows.pop(0) if self.rows else None

    def finish_fast_foundation_depth(self, data_id, *, succeeded):
        self.results.append((data_id, succeeded))
        return True


def setup(root):
    model = root / "model"
    model.mkdir()
    (model / "weights.pth").touch()
    (model / "cfg.yaml").write_text("{}")
    return {"paths": {"fast_foundation_root": str(model), "ckpt": str(model / "weights.pth")},
            "parameters": {"start": 0, "num_frames": "auto", "save_depth_png": True, "save_npy": False}}


def session(root, name):
    path = root / name
    videos = path / "videos"
    videos.mkdir(parents=True)
    for eye in ("left", "right"):
        writer = cv2.VideoWriter(str(videos / f"{eye}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 32))
        assert writer.isOpened()
        for _ in range(3):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()
    manifest = path / "dataset_manifest" / "dataset_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"videos": {eye: {"path": str(videos / f"{eye}.avi")} for eye in ("left", "right")},
        "intrinsics": {"left": {"fx": 50, "fy": 50, "cx": 16, "cy": 16}}, "extrinsics": {"baseline_m": 0.1}}))
    return path


def publish(task, data):
    depth = task["output_dir"]
    pngs = depth / "depth_uint16_png"
    pngs.mkdir(parents=True, exist_ok=True)
    for idx in task["frame_indices"]:
        cv2.imwrite(str(pngs / f"frame_{idx:06d}_depth_mm.png"), np.ones((32, 32), dtype=np.uint16))
    (depth / "fast_foundation_stereo_video_meta.json").write_text(json.dumps({
        "manifest_path": str(task["manifest"]), "start": data["parameters"].get("start", 0),
        "num_frames": None if data["parameters"]["num_frames"] == "auto" else data["parameters"]["num_frames"]}))
    return {"status": "complete"}


def test_success_verifies_requested_frames(tmp_path):
    data = setup(tmp_path)
    session(tmp_path, "session")
    service = Service(["session"])
    result = run_pending(service, tmp_path, data, runner=publish)
    assert result[0]["status"] == "completed"
    assert result[0]["frame_count"] == 3
    assert service.results == [(0, True)]


def test_missing_frame_cannot_complete(tmp_path):
    data = setup(tmp_path)
    session(tmp_path, "session")
    service = Service(["session"])

    def partial(task, data):
        summary = publish(task, data)
        (task["output_dir"] / "depth_uint16_png" / "frame_000001_depth_mm.png").unlink()
        return summary

    assert run_pending(service, tmp_path, data, runner=partial)[0]["status"] == "failed"
    assert service.results == [(0, False)]


def test_range_larger_than_video_does_not_run(tmp_path):
    data = setup(tmp_path)
    data["parameters"]["num_frames"] = 600
    session(tmp_path, "session")
    result = run_pending(Service(["session"]), tmp_path, data, runner=lambda *a: pytest.fail("must not run"))
    assert result[0]["status"] == "failed"


def test_paths_preflight_limit_and_interrupt(tmp_path):
    data = setup(tmp_path)
    service = Service(["../outside", "missing"])
    assert all(r["status"] == "failed" for r in run_pending(service, tmp_path, data, runner=publish))
    session(tmp_path, "session")
    service = Service(["session", "session"])
    run_pending(service, tmp_path, data, limit=1, runner=publish)
    assert len(service.rows) == 1

    def interrupted(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pending(service, tmp_path, data, runner=interrupted)
    assert service.results == [(0, True), (1, False)]
    service = Service(["session"])
    (tmp_path / "model" / "weights.pth").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        run_pending(service, tmp_path, data, runner=publish)
    assert len(service.rows) == 1
