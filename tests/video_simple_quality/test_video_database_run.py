from argparse import Namespace

import pytest

from roboego_hand_vis.video_simple_quality.cli import config_from_yaml
from roboego_hand_vis.video_simple_quality.database_run import run_pending


class Service:
    def __init__(self, paths):
        self.rows = [dict(id=i, local_path=path) for i, path in enumerate(paths)]
        self.results = []

    def claim_video_simple_quality(self):
        return self.rows.pop(0) if self.rows else None

    def finish_video_simple_quality(self, data_id, *, passed):
        self.results.append((data_id, passed))
        return True


def config(root, streams=("left", "right")):
    return config_from_yaml({"parameters": {"camera_streams": list(streams)}}, Namespace(
        session=str(root), video="", output_json=""))


def test_gate_pass_block_and_execution_failure(tmp_path):
    for name in ("pass", "block", "error"):
        (tmp_path / name).mkdir()
    service = Service(["pass", "block", "error"])

    def check(cfg):
        assert cfg.output_json == cfg.session / "annotation" / "video_simple_quality.json"
        assert cfg.video is None
        if cfg.session.name == "error":
            raise RuntimeError("unreadable video")
        return {"passed": cfg.session.name == "pass"}

    results = run_pending(service, config(tmp_path), runner=check)
    assert service.results == [(0, True), (1, False), (2, None)]
    assert [row["status"] for row in results] == ["completed", "block", "failed"]
    assert results[2]["error"] == "unreadable video"


@pytest.mark.parametrize("summary", [{}, {"passed": None}, {"passed": "false"}])
def test_invalid_gate_is_failure(tmp_path, summary):
    (tmp_path / "session").mkdir()
    service = Service(["session"])
    assert run_pending(service, config(tmp_path), runner=lambda _: summary)[0]["status"] == "failed"
    assert service.results == [(0, None)]


def test_unsafe_session_never_runs(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    service = Service(["../outside", "missing", "link"])
    results = run_pending(service, config(tmp_path), runner=lambda _: pytest.fail("must not run"))
    assert all(row["status"] == "failed" for row in results)
    assert service.results == [(0, None), (1, None), (2, None)]


@pytest.mark.parametrize("streams", [(), ("left",), ("left", "right", "bleft")])
def test_both_eyes_required_before_claim(tmp_path, streams):
    service = Service(["session"])
    with pytest.raises(ValueError, match="left and right"):
        run_pending(service, config(tmp_path, streams))
    assert len(service.rows) == 1


def test_interruption_and_writeback_failure(tmp_path):
    (tmp_path / "session").mkdir()
    service = Service(["session"])

    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pending(service, config(tmp_path), runner=interrupted)
    assert service.results == [(0, None)]
    service = Service(["session", "session"])
    service.finish_video_simple_quality = lambda *a, **kw: False
    with pytest.raises(RuntimeError, match="write"):
        run_pending(service, config(tmp_path), runner=lambda _: {"passed": True})
    assert len(service.rows) == 1


def test_batch_limit(tmp_path):
    (tmp_path / "session").mkdir()
    service = Service(["session", "session"])
    assert len(run_pending(service, config(tmp_path), limit=1, runner=lambda _: {"passed": True})) == 1
    assert len(service.rows) == 1
