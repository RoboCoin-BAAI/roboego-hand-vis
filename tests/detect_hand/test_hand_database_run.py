from argparse import Namespace

import pytest

from roboego_hand_vis.detect_hand.cli import config_from_yaml
from roboego_hand_vis.detect_hand.database_run import run_pending


class Service:
    def __init__(self, paths):
        self.rows = [dict(id=i, local_path=path) for i, path in enumerate(paths)]
        self.results = []

    def claim_detect_hand(self):
        return self.rows.pop(0) if self.rows else None

    def finish_detect_hand(self, data_id, *, succeeded):
        self.results.append((data_id, succeeded))
        return True


def config(root):
    model = root / "detector.pt"
    model.touch()
    return config_from_yaml({}, Namespace(video=str(root / "unused.mkv"), model=str(model),
        output_json="", sample_every=None, conf=None, device="cpu", imgsz=None, batch_size=None))


def session(root, name):
    videos = root / name / "videos"
    videos.mkdir(parents=True)
    (videos / "left.mkv").touch()


def test_low_hand_presence_still_completes(tmp_path):
    session(tmp_path, "session")
    service = Service(["session"])

    def check(cfg):
        assert cfg.video == tmp_path / "session" / "videos" / "left.mkv"
        assert cfg.output_json == tmp_path / "session" / "annotation" / "detect_hand_left.json"
        return {"left_hand_presence_percent": 0.0, "right_hand_presence_percent": 1.0}

    result = run_pending(service, tmp_path, config(tmp_path), runner=check)[0]
    assert result["status"] == "completed"
    assert result["summary"]["left_hand_presence_percent"] == 0.0
    assert service.results == [(0, True)]


def test_execution_error_fails_and_continues(tmp_path):
    session(tmp_path, "error")
    session(tmp_path, "good")
    service = Service(["error", "good"])

    def check(cfg):
        if cfg.video.parent.parent.name == "error":
            raise RuntimeError("inference failed")
        return {"sample_count": 10}

    results = run_pending(service, tmp_path, config(tmp_path), runner=check)
    assert [row["status"] for row in results] == ["failed", "completed"]
    assert service.results == [(0, False), (1, True)]


def test_unsafe_or_missing_video_never_runs(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    service = Service(["../outside", "missing", "link"])
    results = run_pending(service, tmp_path, config(tmp_path), runner=lambda _: pytest.fail("must not run"))
    assert all(row["status"] == "failed" for row in results)
    assert service.results == [(0, False), (1, False), (2, False)]


def test_missing_model_rejected_before_claim(tmp_path):
    cfg = config(tmp_path)
    cfg.model.unlink()
    service = Service(["session"])
    with pytest.raises(ValueError, match="model"):
        run_pending(service, tmp_path, cfg)
    assert len(service.rows) == 1


def test_limit_interruption_and_writeback_failure(tmp_path):
    session(tmp_path, "session")
    cfg = config(tmp_path)
    service = Service(["session", "session"])
    run_pending(service, tmp_path, cfg, limit=1, runner=lambda _: {})
    assert len(service.rows) == 1

    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pending(service, tmp_path, cfg, runner=interrupted)
    assert service.results == [(0, True), (1, False)]
    service = Service(["session", "session"])
    service.finish_detect_hand = lambda *a, **kw: False
    with pytest.raises(RuntimeError, match="write"):
        run_pending(service, tmp_path, cfg, runner=lambda _: {})
    assert len(service.rows) == 1
