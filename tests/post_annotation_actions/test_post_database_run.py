import json

import pytest

from roboego_hand_vis.post_annotation_actions.database_run import run_pending


class Service:
    def __init__(self, paths):
        self.rows = [dict(id=i, local_path=path) for i, path in enumerate(paths)]
        self.results = []

    def claim_post_annotation_actions(self):
        return self.rows.pop(0) if self.rows else None

    def finish_post_annotation_actions(self, data_id, *, succeeded):
        self.results.append((data_id, succeeded))
        return True


def setup(root, monkeypatch):
    monkeypatch.setenv("TEST_POST_KEY", "test-only")
    return {"parameters": {"api_key_env": "TEST_POST_KEY", "generate_subtasks": False,
                           "use_vl_object_resolver": False}}


def session(root, name):
    chunks = root / name / "annotation" / "qwen_frame_annotation_chunks"
    chunks.mkdir(parents=True)
    (chunks / "chunk_0000.json").write_text(json.dumps({"frames": [{"t": 0}]}))


def completed(cfg):
    cfg.resolved_chunk_dir.mkdir(parents=True, exist_ok=True)
    (cfg.resolved_chunk_dir / "chunk_0000.json").write_text(json.dumps({"frames": [{"t": 0}]}))
    cfg.merged_output_json.write_text(json.dumps({"frame_count": 1, "frames": [{"t": 0}]}))
    return {"status": "complete", "merged_frame_count": 1}


def test_success_paths_and_overwrite(tmp_path, monkeypatch):
    data = setup(tmp_path, monkeypatch)
    session(tmp_path, "first")
    session(tmp_path, "second")
    service = Service(["first", "second"])
    configs = []

    def check(cfg):
        assert cfg.overwrite is True
        annotation = cfg.qwen_chunk_dir.parent
        assert cfg.qwen_chunk_dir.name == "qwen_frame_annotation_chunks"
        for path in (cfg.action_chunk_dir, cfg.review_chunk_dir, cfg.resolved_chunk_dir,
                     cfg.merged_output_json, cfg.subtask_output_json, cfg.frames_dir):
            assert path.parent == annotation
        configs.append(cfg)
        return completed(cfg)

    results = run_pending(service, tmp_path, data, overwrite=True, runner=check)
    assert all(row["status"] == "completed" for row in results)
    assert service.results == [(0, True), (1, True)]
    assert configs[0].resolved_chunk_dir != configs[1].resolved_chunk_dir


def test_missing_chunks_or_unsafe_path_do_not_run(tmp_path, monkeypatch):
    data = setup(tmp_path, monkeypatch)
    service = Service(["missing", "../outside"])
    results = run_pending(service, tmp_path, data, runner=lambda _: pytest.fail("must not run"))
    assert all(row["status"] == "failed" for row in results)
    session(tmp_path, "empty")
    (tmp_path / "empty" / "annotation" / "qwen_frame_annotation_chunks" / "chunk_0000.json").unlink()
    assert run_pending(Service(["empty"]), tmp_path, data, runner=lambda _: pytest.fail("must not run"))[0]["status"] == "failed"


@pytest.mark.parametrize("mode", ["missing", "vl_failed", "zero_frames", "subtasks_missing"])
def test_incomplete_outputs_cannot_complete(tmp_path, monkeypatch, mode):
    data = setup(tmp_path, monkeypatch)
    if mode == "subtasks_missing":
        prompt = tmp_path / "prompt.md"
        prompt.write_text("test")
        data["parameters"].update(generate_subtasks=True, subtask_prompt=str(prompt), subtask_object_prompt=str(prompt))
    session(tmp_path, "session")
    service = Service(["session"])

    def check(cfg):
        if mode == "missing":
            return {"status": "complete", "merged_frame_count": 1}
        summary = completed(cfg)
        if mode == "vl_failed":
            (cfg.resolved_chunk_dir / "chunk_0000.json").write_text(json.dumps(
                {"frames": [{"t": 0}], "vl_object_resolver_status": "parse_failed"}))
        if mode == "zero_frames":
            summary["merged_frame_count"] = 0
        return summary

    assert run_pending(service, tmp_path, data, runner=check)[0]["status"] == "failed"
    assert service.results == [(0, False)]


def test_preflight_before_claim(tmp_path, monkeypatch):
    data = setup(tmp_path, monkeypatch)
    service = Service(["session"])
    monkeypatch.delenv("TEST_POST_KEY")
    with pytest.raises(ValueError, match="TEST_POST_KEY"):
        run_pending(service, tmp_path, data)
    monkeypatch.setenv("TEST_POST_KEY", "test-only")
    data["parameters"]["resolved_chunk_dir"] = str(tmp_path / "shared")
    with pytest.raises(ValueError, match="resolved_chunk_dir"):
        run_pending(service, tmp_path, data)
    assert len(service.rows) == 1


def test_exception_interruption_limit_and_writeback(tmp_path, monkeypatch):
    data = setup(tmp_path, monkeypatch)
    session(tmp_path, "session")
    service = Service(["session", "session"])

    def failed(_):
        raise RuntimeError("API failed")

    assert run_pending(service, tmp_path, data, limit=1, runner=failed)[0]["status"] == "failed"
    assert len(service.rows) == 1

    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pending(service, tmp_path, data, runner=interrupted)
    assert service.results == [(0, False), (1, False)]
    service = Service(["session", "session"])
    service.finish_post_annotation_actions = lambda *a, **kw: False
    with pytest.raises(RuntimeError, match="write"):
        run_pending(service, tmp_path, data, runner=completed)
    assert len(service.rows) == 1
