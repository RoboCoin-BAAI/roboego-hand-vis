"""Object segmentation pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from roboego_hand_vis.pipelines.objects.nodes import run_sam3_hand_object_segmentation_node


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_sam3_hand_object_segmentation_node,
                inputs={"config": "params:sam3_hand_object_segmentation"},
                outputs="sam3_hand_object_segmentation_report",
                name="sam3_hand_object_segmentation",
                tags=["objects", "sam3"],
            ),
        ]
    )
