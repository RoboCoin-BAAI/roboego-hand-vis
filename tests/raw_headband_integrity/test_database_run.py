from argparse import Namespace

import pytest

from roboego_hand_vis.raw_headband_integrity.cli import config_from_yaml
from roboego_hand_vis.raw_headband_integrity.database_run import run_pending


class Service:
    def __init__(self, paths):
        self.rows = [dict(id=i, local_path=path) for i, path in enumerate(paths)]
        self.results = []

    def claim_raw_headband_integrity(self):
        return self.rows.pop(0) if self.rows else None

    def finish_raw_headband_integrity(self, data_id, *, passed):
        self.results.append((data_id, passed))
        return True


def config(root, enabled=True):
    return config_from_yaml({"gate": {"enabled": enabled}}, Namespace(
        session=str(root), output_root="", camera_stream=None, overwrite=True))


def test_gate_failure_and_execution_error_are_failed(tmp_path):
    for name in ("pass", "fail", "error"):
        (tmp_path / name).mkdir()
    service = Service(["pass", "fail", "error"])

    def check(cfg):
        assert cfg.output_root == cfg.session
        if cfg.session.name == "error":
            raise RuntimeError("bad input")
        return {"gate": {"passed": cfg.session.name == "pass"}}

    results = run_pending(service, config(tmp_path), runner=check)
    assert service.results == [(0, True), (1, False), (2, None)]
    assert [row["status"] for row in results] == ["completed", "block", "failed"]
    assert results[2]["error"] == "bad input"


def test_unsafe_or_missing_session_never_runs(tmp_path):
    service = Service(["../outside", "missing"])
    results = run_pending(service, config(tmp_path), runner=lambda _: pytest.fail("must not run"))
    assert service.results == [(0, None), (1, None)]
    assert all(row["status"] == "failed" for row in results)


def test_disabled_gate_rejected_before_claim(tmp_path):
    service = Service(["session"])
    with pytest.raises(ValueError, match="gate"):
        run_pending(service, config(tmp_path, False))
    assert len(service.rows) == 1


def test_interruption_marks_failed_and_propagates(tmp_path):
    (tmp_path / "session").mkdir()
    service = Service(["session"])

    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pending(service, config(tmp_path), runner=interrupted)
    assert service.results == [(0, None)]


@pytest.mark.parametrize("summary", [{}, {"gate": {"passed": None}}, {"gate": {"passed": "false"}}])
def test_missing_or_invalid_gate_result_is_execution_failure(tmp_path, summary):
    (tmp_path / "session").mkdir()
    service = Service(["session"])
    assert run_pending(service, config(tmp_path), runner=lambda _: summary)[0]["status"] == "failed"
    assert service.results == [(0, None)]


def test_limit_and_writeback_failure(tmp_path):
    (tmp_path / "session").mkdir()
    service = Service(["session", "session"])
    run_pending(service, config(tmp_path), limit=1, runner=lambda _: {"gate": {"passed": True}})
    assert len(service.rows) == 1
    service.finish_raw_headband_integrity = lambda *a, **kw: False
    with pytest.raises(RuntimeError, match="write"):
        run_pending(service, config(tmp_path), runner=lambda _: {"gate": {"passed": True}})
