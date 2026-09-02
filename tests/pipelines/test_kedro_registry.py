from kedro.pipeline import Pipeline

from roboego_hand_vis.pipeline_registry import register_pipelines


def test_register_pipelines_exposes_stage_boundaries():
    pipelines = register_pipelines()

    assert set(pipelines) == {
        "__default__",
        "datasets",
        "quality",
        "semantic",
        "depth",
        "egohand",
        "restoration",
        "objects",
        "export",
        "experiments",
    }
    assert all(isinstance(pipeline, Pipeline) for pipeline in pipelines.values())


def test_stage_pipelines_have_smoke_nodes_for_cli_execution():
    pipelines = register_pipelines()

    for name, pipeline in pipelines.items():
        if name == "__default__":
            continue
        expected_node_count = {"datasets": 6, "quality": 7, "semantic": 6, "depth": 3, "egohand": 10}.get(name, 1)
        assert len(pipeline.nodes) == expected_node_count


def test_default_pipeline_combines_all_stage_smoke_nodes():
    pipelines = register_pipelines()

    assert len(pipelines["__default__"].nodes) == 36
