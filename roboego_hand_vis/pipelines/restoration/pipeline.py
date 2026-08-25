"""Restoration pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from roboego_hand_vis.pipelines.restoration.nodes import run_realvdeblur_stereo_node


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_realvdeblur_stereo_node,
                inputs="params:realvdeblur_stereo",
                outputs="realvdeblur_stereo_report",
                name="realvdeblur_stereo",
                tags=["restoration", "realvdeblur"],
            ),
        ]
    )
