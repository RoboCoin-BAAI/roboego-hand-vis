"""Privacy pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from roboego_hand_vis.pipelines.privacy.nodes import run_egoblur_temporal_face_blur_node


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_egoblur_temporal_face_blur_node,
                inputs="params:egoblur_temporal_face_blur",
                outputs="egoblur_temporal_face_blur_report",
                name="egoblur_temporal_face_blur",
                tags=["privacy", "egoblur"],
            ),
        ]
    )
