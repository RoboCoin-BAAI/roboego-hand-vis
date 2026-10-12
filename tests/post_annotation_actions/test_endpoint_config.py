from roboego_hand_vis.post_annotation_actions.cli import config_from_yaml
from roboego_hand_vis.post_annotation_actions import robolabel_actions


def test_url_config_environment_and_default(tmp_path, monkeypatch):
    url = "http://model.example/v1/chat/completions"
    monkeypatch.setenv("DASHSCOPE_BASE_URL", url)
    assert config_from_yaml({}, qwen_chunk_dir=tmp_path).base_url == url
    assert config_from_yaml({"parameters": {"base_url": "  "}}, qwen_chunk_dir=tmp_path).base_url == url
    explicit = "http://other.example/v1/chat/completions"
    assert config_from_yaml({"parameters": {"base_url": explicit}}, qwen_chunk_dir=tmp_path).base_url == explicit
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "  ")
    assert config_from_yaml({"parameters": {"base_url": "  "}}, qwen_chunk_dir=tmp_path).base_url == "https://llm-g3o8d3j71xbf6prc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
    monkeypatch.delenv("DASHSCOPE_BASE_URL")
    assert config_from_yaml({}, qwen_chunk_dir=tmp_path).base_url == "https://llm-g3o8d3j71xbf6prc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"


def test_visual_resolver_receives_configured_endpoint(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(robolabel_actions, "build_vl_object_resolver_payload", lambda *a, **kw: {"frames": [{"t": 0}]})
    def resolve(**kwargs):
        seen.append(kwargs["base_url"])
        return {"status": "ok", "frames": []}
    robolabel_actions.standardize_stage1_actions(
        {"frames": []}, frames_dir=tmp_path, use_vl_object_resolver=True,
        vl_object_resolver=resolve, base_url="http://model.example/v1/chat/completions")
    assert seen == ["http://model.example/v1/chat/completions"]


def test_stage_defaults_use_requested_model(tmp_path):
    from roboego_hand_vis.post_annotation_actions.cli import DEFAULT_CONFIG, load_yaml
    for data in ({}, load_yaml(DEFAULT_CONFIG)):
        config = config_from_yaml(data, qwen_chunk_dir=tmp_path)
        assert (config.text_model, config.subtask_model, config.vl_object_model) == ('qwen3.8-27b',) * 3
    assert config_from_yaml(load_yaml(DEFAULT_CONFIG), qwen_chunk_dir=tmp_path).vl_object_model == 'qwen3.8-27b'
    assert config_from_yaml({'parameters': {'vl_object_model': 'custom-vl'}}, qwen_chunk_dir=tmp_path).vl_object_model == 'custom-vl'
