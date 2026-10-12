import json

import pytest
import yaml

from roboego_hand_vis.qwen_annotation.frame_annotation import (
    build_data_inspection_placeholder_frames,
    call_qwen_frame_annotations,
)


def make_session(tmp_path):
    video = tmp_path / "videos" / "left.mkv"
    video.parent.mkdir()
    video.touch()
    (tmp_path / "任务详情.txt").write_text(
        "日期：2026/09/23\n采集人：某人\n任务名称：收纳砂纸\n"
        "任务内容：1.将不同粒度的砂纸对齐堆叠\n\t2.整齐放入收纳格\n任务道具：砂纸，收纳盒",
        encoding="utf-8",
    )
    (tmp_path / "厨房灶台_收纳砂纸_config.yaml").write_text(yaml.safe_dump({
        "task": "收纳砂纸", "scene": {
            "name": "厨房灶台", "notes": "1.将不同粒度的砂纸对齐堆叠\n2.整齐放入收纳格",
            "opts": {"renderer": "webgl"}, "objects": [
                {"name": "砂纸", "lib_name": "砂纸", "material": "纸质"},
                {"name": "砂纸_c", "lib_name": "砂纸", "material": "纸质"},
                {"name": "砂纸_c2", "lib_name": "砂纸", "material": "纸质"},
                {"name": "黄色收纳盒", "lib_name": "黄色收纳盒", "material": "塑料"},
            ],
        },
    }, allow_unicode=True), encoding="utf-8")
    return video


def translation():
    return {
        "scene": "kitchen stovetop",
        "task_description": "Align and stack sandpaper sheets of different grit sizes, then place them neatly into the storage compartment.",
        "objects": [
            {"source_name": "砂纸", "name": "sandpaper"},
            {"source_name": "黄色收纳盒", "name": "yellow plastic storage box"},
        ],
    }


def frame():
    return build_data_inspection_placeholder_frames([0.0], {})[0]


def test_extracts_only_semantic_context_and_merges_library_names(tmp_path):
    from roboego_hand_vis.qwen_annotation.task_context import load_task_context
    source = load_task_context(make_session(tmp_path))
    assert source["scene"] == "厨房灶台"
    assert "不同粒度" in source["task_description"]
    assert "整齐放入收纳格" in source["task_description"]
    assert [o["name"] for o in source["objects"]] == ["砂纸", "黄色收纳盒"]
    assert source["objects"][1]["material"] == "塑料"
    assert len(source["sources"]) == 2
    assert "webgl" not in json.dumps(source)
    assert "某人" not in json.dumps(source, ensure_ascii=False)


def test_missing_and_partial_metadata(tmp_path):
    from roboego_hand_vis.qwen_annotation.task_context import load_task_context
    video = make_session(tmp_path)
    (tmp_path / "厨房灶台_收纳砂纸_config.yaml").unlink()
    source = load_task_context(video)
    assert source["scene"] == ""
    assert [o["name"] for o in source["objects"]] == ["砂纸", "收纳盒"]
    (tmp_path / "任务详情.txt").unlink()
    assert load_task_context(video) is None


def test_yaml_only_standalone_video_and_ambiguous_configs(tmp_path):
    from roboego_hand_vis.qwen_annotation.task_context import load_task_context
    make_session(tmp_path)
    (tmp_path / "任务详情.txt").unlink()
    assert "整齐放入" in load_task_context(tmp_path / "left.mkv")["task_description"]
    (tmp_path / "second_config.yaml").write_text("{}")
    with pytest.raises(ValueError, match="multiple"):
        load_task_context(tmp_path / "left.mkv")


@pytest.mark.parametrize("value", ["[", "[]", "scene: invalid", "scene: {objects: invalid}"])
def test_invalid_metadata_is_not_silently_ignored(tmp_path, value):
    from roboego_hand_vis.qwen_annotation.task_context import load_task_context
    (tmp_path / "x_config.yaml").write_text(value)
    with pytest.raises(ValueError):
        load_task_context(tmp_path / "left.mkv")


def test_translation_schema_requires_complete_unique_english_mapping(tmp_path):
    from roboego_hand_vis.qwen_annotation.task_context import load_task_context, parse_task_translation
    source = load_task_context(make_session(tmp_path))
    assert parse_task_translation(json.dumps(translation()), source)["status"] == "ok"
    for mutation in (
        lambda x: x.update(scene="厨房灶台"),
        lambda x: x.update(task_description=""),
        lambda x: x["objects"].pop(),
        lambda x: x["objects"].append({"source_name": "手机", "name": "phone"}),
        lambda x: x["objects"][1].update(name="sandpaper"),
        lambda x: x["objects"][0].update(source_name="砂纸_c"),
    ):
        value = translation()
        mutation(value)
        assert parse_task_translation(json.dumps(value), source)["status"] == "validation_failed"


@pytest.mark.parametrize("field", ["task_objects", "attention_objects", "interaction_objects", "interaction_object_attributes"])
def test_rejects_objects_outside_vocabulary(field):
    from roboego_hand_vis.qwen_annotation.task_context import validate_task_objects
    item = frame()
    item[field] = [{"name": "phone"}] if field.endswith("attributes") else ["phone"]
    with pytest.raises(ValueError, match="phone"):
        validate_task_objects([item], translation())


def test_allowed_objects_and_empty_observations_are_accepted():
    from roboego_hand_vis.qwen_annotation.task_context import validate_task_objects
    item = frame()
    validate_task_objects([item], translation())
    item["task_objects"] = ["sandpaper", "yellow plastic storage box"]
    validate_task_objects([item], translation())
    item["interaction_objects"] = ["sandpaper"]
    item["interaction_object_attributes"] = [{"name": "sandpaper"}]
    item["activity"] = "wearer holds a sheet"
    with pytest.raises(ValueError, match="activity"):
        validate_task_objects([item], translation())


def test_client_retries_custom_translation_validation(monkeypatch, tmp_path):
    from roboego_hand_vis.qwen_annotation.task_context import load_task_context, parse_task_translation
    source = load_task_context(make_session(tmp_path))
    invalid = translation()
    invalid["objects"].pop()
    responses = [invalid, translation()]
    requests = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps(responses.pop(0))}}]}).encode()

    def urlopen(req, timeout):
        requests.append(json.loads(req.data))
        return Response()

    monkeypatch.setenv("TEST_QWEN_KEY", "test")
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result = call_qwen_frame_annotations(
        messages=[{"role": "user", "content": "translate"}], model="test", api_key_env="TEST_QWEN_KEY",
        max_retries=1, retry_sleep_s=0,
        response_parser=lambda text: parse_task_translation(text, source),
    )
    assert result["status"] == "ok"
    assert result["retry_count"] == 1
    assert "Validation error" in requests[1]["messages"][0]["content"][-1]["text"]


def setup_run(tmp_path, monkeypatch, *, with_metadata=True):
    from roboego_hand_vis.qwen_annotation import cli, compute
    if with_metadata:
        video = make_session(tmp_path)
    else:
        video = tmp_path / "videos" / "left.mkv"
        video.parent.mkdir()
        video.touch()
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake image")
    monkeypatch.setattr(compute, "extract_video_frames", lambda **kw: ([image, image], [0.0, 1.0]))
    monkeypatch.setenv("TEST_QWEN_KEY", "test")
    config = cli.config_from_yaml({"parameters": {
        "api_key_env": "TEST_QWEN_KEY", "batch_size": 1, "retry_sleep_s": 0, "max_retries": 1,
    }}, video=video, output_json=tmp_path / "annotation" / "qwen.json")
    requests = []
    replies = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": json.dumps(replies.pop(0))}}],
                "usage": {"total_tokens": 10},
            }).encode()

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return config, requests, replies


def test_run_translates_once_constrains_every_batch_and_persists_metadata(tmp_path, monkeypatch):
    from roboego_hand_vis.qwen_annotation.compute import run_qwen_annotation
    config, requests, replies = setup_run(tmp_path, monkeypatch)
    second = frame()
    second["t"] = 1.0
    replies.extend([translation(), {"frames": [frame()]}, {"frames": [second]}])
    result = run_qwen_annotation(config)
    assert result["status"] == "complete"
    assert len(requests) == 3
    for request in requests[1:]:
        assert "CLOSED vocabulary" in request["messages"][0]["content"]
        content = request["messages"][1]["content"][0]["text"]
        assert translation()["task_description"] in content
        assert "kitchen stovetop" in content
        assert "yellow plastic storage box" in content
    payload = json.loads(config.output_json.read_text())
    assert payload["task_metadata"]["scene"] == "kitchen stovetop"
    assert payload["task_metadata"]["translation_usage"]["total_tokens"] == 10
    assert payload["usage"]["total_tokens"] == 20
    for path in [config.checkpoint_json, *config.batch_output_dir.glob("*.json"), *config.chunk_output_dir.glob("*.json")]:
        assert json.loads(path.read_text())["task_metadata"] == payload["task_metadata"]
    run_qwen_annotation(config)
    assert len(requests) == 3


@pytest.mark.parametrize("mutation", ["source", "removed", "cache_missing", "prompt"])
def test_changed_context_cannot_reuse_previous_artifacts(tmp_path, monkeypatch, mutation):
    from dataclasses import replace
    from roboego_hand_vis.qwen_annotation.compute import run_qwen_annotation
    config, requests, replies = setup_run(tmp_path, monkeypatch)
    replies.extend([translation(), {"frames": [frame()]}, {"frames": [frame()]}])
    run_qwen_annotation(config)
    original = config.output_json.read_bytes()
    if mutation == "source":
        (tmp_path / "任务详情.txt").write_text("任务名称：新任务\n任务道具：砂纸")
    elif mutation == "removed":
        (tmp_path / "任务详情.txt").unlink()
        (tmp_path / "厨房灶台_收纳砂纸_config.yaml").unlink()
    elif mutation == "cache_missing":
        (config.raw_output_dir / "task_context.json").unlink()
    else:
        prompt = tmp_path / "changed_prompt.md"
        prompt.write_text(config.prompt.read_text() + "\n")
        config = replace(config, prompt=prompt)
    with pytest.raises(ValueError, match="fresh output"):
        run_qwen_annotation(config)
    assert len(requests) == 3
    assert config.output_json.read_bytes() == original


@pytest.mark.parametrize("artifact", ["checkpoint", "raw", "batch", "chunk", "output"])
def test_legacy_artifacts_rejected_before_translation(tmp_path, monkeypatch, artifact):
    from roboego_hand_vis.qwen_annotation.compute import run_qwen_annotation
    config, requests, _ = setup_run(tmp_path, monkeypatch)
    path = {
        "checkpoint": config.checkpoint_json,
        "raw": config.raw_output_dir / "batch_0000.raw.txt",
        "batch": config.batch_output_dir / "batch_0000.json",
        "chunk": config.chunk_output_dir / "chunk_0000.json",
        "output": config.output_json,
    }[artifact]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="fresh output"):
        run_qwen_annotation(config)
    assert not requests
    assert path.read_text() == "{}"


def test_without_metadata_retains_legacy_flow(tmp_path, monkeypatch):
    from roboego_hand_vis.qwen_annotation.compute import run_qwen_annotation
    config, requests, replies = setup_run(tmp_path, monkeypatch, with_metadata=False)
    replies.extend([{"frames": [frame()]}, {"frames": [frame()]}])
    assert run_qwen_annotation(config)["status"] == "complete"
    assert len(requests) == 2
    assert "task_metadata" not in json.loads(config.output_json.read_text())


def test_outside_vocabulary_is_retried_and_never_saved_as_success(tmp_path, monkeypatch):
    from roboego_hand_vis.qwen_annotation.compute import run_qwen_annotation
    config, requests, replies = setup_run(tmp_path, monkeypatch)
    bad = frame()
    bad["task_objects"] = ["phone"]
    replies.extend([translation(), {"frames": [bad]}, {"frames": [bad]}, {"frames": [frame()]}])
    result = run_qwen_annotation(config)
    assert result["status"] == "partial_failed"
    assert result["failed_batch_count"] == 1
    assert "phone" in requests[2]["messages"][1]["content"][-1]["text"]
    assert all("phone" not in f["task_objects"] for f in json.loads(config.output_json.read_text())["frames"])


def test_recovery_rechecks_vocabulary(tmp_path, monkeypatch):
    from dataclasses import replace
    from roboego_hand_vis.qwen_annotation.compute import run_qwen_annotation
    config, requests, replies = setup_run(tmp_path, monkeypatch)
    replies.extend([translation(), {"frames": [frame()]}, {"frames": [frame()]}])
    run_qwen_annotation(config)
    bad = frame()
    bad["task_objects"] = ["phone"]
    (config.raw_output_dir / "batch_0000.raw.txt").write_text(json.dumps({"frames": [bad]}))
    replies.append({"frames": [frame()]})
    result = run_qwen_annotation(replace(config, recover_from_raw=True))
    assert result["status"] == "complete"
    assert len(requests) == 4
    assert all("phone" not in f["task_objects"] for f in json.loads(config.output_json.read_text())["frames"])


def test_task_steps_can_contain_colons(tmp_path):
    from roboego_hand_vis.qwen_annotation.task_context import load_task_context
    (tmp_path / "任务详情.txt").write_text(
        "任务名称：收纳砂纸\n任务内容：步骤1：对齐砂纸\n步骤2：整齐放入收纳格\n任务道具：砂纸，收纳盒"
    )
    assert "步骤2：整齐放入收纳格" in load_task_context(tmp_path / "left.mkv")["task_description"]


def test_translation_failure_stops_before_extracting_frames(tmp_path, monkeypatch):
    from roboego_hand_vis.qwen_annotation import compute
    config, requests, replies = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(compute, "extract_video_frames", lambda **kw: pytest.fail("must not extract frames"))
    replies.extend([{"objects": []}, {"objects": []}])
    with pytest.raises(RuntimeError, match="translation failed"):
        compute.run_qwen_annotation(config)
    assert len(requests) == 2
    assert not config.output_json.exists()
    assert not (config.raw_output_dir / "task_context.json").exists()


def test_checkpoint_objects_revalidated_on_resume(tmp_path, monkeypatch):
    from roboego_hand_vis.qwen_annotation.compute import run_qwen_annotation
    config, requests, replies = setup_run(tmp_path, monkeypatch)
    replies.extend([translation(), {"frames": [frame()]}, {"frames": [frame()]}])
    run_qwen_annotation(config)
    checkpoint = json.loads(config.checkpoint_json.read_text())
    checkpoint["frames"][0]["task_objects"] = ["phone"]
    config.checkpoint_json.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="phone"):
        run_qwen_annotation(config)
    assert len(requests) == 3


@pytest.mark.parametrize("field", ["scene", "activity", "notes"])
def test_context_frames_reject_chinese_descriptions(field):
    from roboego_hand_vis.qwen_annotation.task_context import parse_context_frame_response
    item = frame()
    item[field] = "厨房灶台上整理砂纸"
    result = parse_context_frame_response(json.dumps({"frames": [item]}), translation())
    assert result["status"] == "validation_failed"
    assert field in result["error"]
