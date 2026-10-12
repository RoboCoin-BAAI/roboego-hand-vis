import json

import pytest

from roboego_hand_vis.stereo_depth_gate.database_run import run_pending


class Service:
    def __init__(self, paths):
        self.rows = [dict(id=i, local_path=p) for i, p in enumerate(paths)]
        self.results = []

    def claim_stereo_depth_gate(self):
        return self.rows.pop(0) if self.rows else None

    def finish_stereo_depth_gate(self, data_id, *, passed):
        self.results.append((data_id, passed))
        return True


def session(root, name):
    path = root / name / "dataset_manifest"
    path.mkdir(parents=True)
    videos = path.parent / "videos"
    videos.mkdir()
    for eye in ("left", "right"):
        (videos / f"{eye}.mkv").write_bytes(b"video")
    (path / "dataset_manifest.json").write_text(json.dumps({"videos": {eye: {"path": str(videos / f"{eye}.mkv")} for eye in ("left", "right")}}))


def test_usable_for_depth_is_gate_even_if_strict_checks_fail(tmp_path):
    for name in ("risk", "block", "error"):
        session(tmp_path, name)
    service = Service(["risk", "block", "error"])

    def check(cfg):
        name = cfg.manifest_path.parent.parent.name
        if name == "error":
            raise RuntimeError("bad stereo")
        return {"usable_for_depth": name == "risk", "summary_passed": False}

    results = run_pending(service, tmp_path, {}, runner=check)
    assert [r["status"] for r in results] == ["completed", "block", "failed"]
    assert service.results == [(0, True), (1, False), (2, None)]


def test_bad_result_and_paths_fail(tmp_path):
    session(tmp_path, "session")
    service = Service(["../outside", "missing", "session"])
    results = run_pending(service, tmp_path, {}, runner=lambda _: {})
    assert all(r["status"] == "failed" for r in results)
    assert service.results == [(0, None), (1, None), (2, None)]
