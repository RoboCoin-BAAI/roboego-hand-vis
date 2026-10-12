import json

import pytest

from roboego_hand_vis.qwen_annotation.database_run import run_pending


class Service:
    def __init__(self, paths):
        self.rows = [dict(id=i, local_path=path) for i, path in enumerate(paths)]
        self.results = []

    def claim_qwen_annotation(self):
        return self.rows.pop(0) if self.rows else None

    def finish_qwen_annotation(self, data_id, *, succeeded):
        self.results.append((data_id, succeeded))
        return True


def setup(root, monkeypatch):
    prompt = root / "prompt.md"
    prompt.write_text("test prompt")
    monkeypatch.setenv("TEST_QWEN_KEY", "test-only")
    return {"parameters": {"prompt": str(prompt), "api_key_env": "TEST_QWEN_KEY"}}


def session(root, name):
    videos = root / name / "videos"
    videos.mkdir(parents=True)
    (videos / "left.mkv").touch()


def completed(cfg):
    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_json.write_text(json.dumps(dict(status="complete", source_video=str(cfg.video),
        failed_batch_count=0, failed_batches=[], frames=[{"t": 0}])))
    return dict(status="complete", failed_batch_count=0, frame_count=1)


def test_success_uses_session_paths_and_preserves_resume(tmp_path, monkeypatch):
    data = setup(tmp_path, monkeypatch)
    for name in ("first", "second"):
        session(tmp_path, name)
    service = Service(["first", "second"])
    configs = []

    def check(cfg):
        assert cfg.video.name == "left.mkv"
        assert cfg.output_json.name == "qwen_frame_annotations_left.json"
        assert cfg.resume is True
        for path in (cfg.frames_dir, cfg.raw_output_dir, cfg.batch_output_dir,
                     cfg.chunk_output_dir, cfg.checkpoint_json):
            assert path.parent == cfg.output_json.parent
        configs.append(cfg)
        return completed(cfg)

    results = run_pending(service, tmp_path, data, runner=check)
    assert [r["status"] for r in results] == ["completed", "completed"]
    assert configs[0].chunk_output_dir != configs[1].chunk_output_dir
    assert service.results == [(0, True), (1, True)]


@pytest.mark.parametrize("summary", [
    {"status": "partial_failed", "failed_batch_count": 1, "frame_count": 1},
    {"status": "failed", "failed_batch_count": 1, "frame_count": 0},
    {"status": "complete", "failed_batch_count": 0, "frame_count": 0}, {},
])
def test_partial_or_invalid_result_is_failed(tmp_path, monkeypatch, summary):
    data = setup(tmp_path, monkeypatch)
    session(tmp_path, "session")
    service = Service(["session"])

    def check(cfg):
        completed(cfg)
        return summary

    assert run_pending(service, tmp_path, data, runner=check)[0]["status"] == "failed"
    assert service.results == [(0, False)]


def test_missing_report_is_failed(tmp_path, monkeypatch):
    data = setup(tmp_path, monkeypatch)
    session(tmp_path, "session")
    service = Service(["session"])
    result = run_pending(service, tmp_path, data, runner=lambda _: dict(status="complete", failed_batch_count=0, frame_count=1))
    assert result[0]["status"] == "failed"


@pytest.mark.parametrize("changes", [
    {"status": "partial_failed", "failed_batch_count": 1},
    {"source_video": "/some/other/video.mkv"}, {"frames": []},
])
def test_report_must_match_success_summary(tmp_path, monkeypatch, changes):
    data = setup(tmp_path, monkeypatch)
    session(tmp_path, "session")
    service = Service(["session"])

    def check(cfg):
        summary = completed(cfg)
        payload = json.loads(cfg.output_json.read_text())
        payload.update(changes)
        cfg.output_json.write_text(json.dumps(payload))
        return summary

    assert run_pending(service, tmp_path, data, runner=check)[0]["status"] == "failed"
    assert service.results == [(0, False)]


def test_credential_and_shared_output_preflight(tmp_path, monkeypatch):
    data = setup(tmp_path, monkeypatch)
    service = Service(["session"])
    monkeypatch.delenv("TEST_QWEN_KEY")
    with pytest.raises(ValueError, match="TEST_QWEN_KEY"):
        run_pending(service, tmp_path, data)
    assert len(service.rows) == 1
    monkeypatch.setenv("TEST_QWEN_KEY", "test-only")
    data["parameters"]["chunk_output_dir"] = str(tmp_path / "shared_chunks")
    with pytest.raises(ValueError, match="chunk_output_dir"):
        run_pending(service, tmp_path, data)
    assert len(service.rows) == 1


def test_error_paths_interruption_limit_and_writeback(tmp_path, monkeypatch):
    data = setup(tmp_path, monkeypatch)
    service = Service(["../outside", "missing"])
    results = run_pending(service, tmp_path, data, runner=lambda _: pytest.fail("must not run"))
    assert all(row["status"] == "failed" for row in results)
    session(tmp_path, "session")
    service = Service(["session", "session"])
    run_pending(service, tmp_path, data, limit=1, runner=completed)
    assert len(service.rows) == 1

    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pending(service, tmp_path, data, runner=interrupted)
    assert service.results == [(0, True), (1, False)]
    service = Service(["session", "session"])
    service.finish_qwen_annotation = lambda *a, **kw: False
    with pytest.raises(RuntimeError, match="write"):
        run_pending(service, tmp_path, data, runner=completed)
    assert len(service.rows) == 1
