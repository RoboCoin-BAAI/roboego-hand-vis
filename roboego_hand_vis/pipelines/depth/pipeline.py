"""Depth pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from roboego_hand_vis.pipelines.depth.nodes import (
    run_archive_depth_cache_node,
    run_fast_foundation_stereo_depth_node,
    run_foundation_stereo_depth_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_foundation_stereo_depth_node,
                inputs="params:foundation_stereo_depth",
                outputs="foundation_stereo_depth_report",
                name="foundation_stereo_depth",
                tags=["depth", "foundation_stereo"],
            ),
            node(
                func=run_fast_foundation_stereo_depth_node,
                inputs="params:fast_foundation_stereo_depth",
                outputs="fast_foundation_stereo_depth_report",
                name="fast_foundation_stereo_depth",
                tags=["depth", "foundation_stereo", "fast_foundation_stereo"],
            ),
            node(
                func=run_archive_depth_cache_node,
                inputs="params:archive_depth_cache",
                outputs="archive_depth_cache_report",
                name="archive_depth_cache",
                tags=["depth", "archive"],
            ),
        ]
    )
