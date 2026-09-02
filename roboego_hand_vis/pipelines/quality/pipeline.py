"""Quality pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from roboego_hand_vis.pipelines.quality.nodes import (
    run_depth_numeric_quality_node,
    run_depth_quality_node_from_config,
    run_pre_semantic_quality_gate_node,
    run_raw_headband_integrity_check,
    run_raw_headband_quality_gate,
    run_stereo_segment_quality_node,
    run_stereo_summary_quality_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_raw_headband_integrity_check,
                inputs="params:raw_headband_quality",
                outputs="raw_headband_quality_report",
                name="raw_headband_integrity_check",
                tags=["quality", "raw_headband"],
            ),
            node(
                func=run_raw_headband_quality_gate,
                inputs={
                    "integrity_result": "raw_headband_quality_report",
                    "config": "params:raw_headband_quality_gate",
                },
                outputs="raw_headband_quality_gate_result",
                name="raw_headband_quality_gate",
                tags=["quality", "raw_headband"],
            ),
            node(
                func=run_pre_semantic_quality_gate_node,
                inputs={
                    "raw_gate_result": "raw_headband_quality_gate_result",
                    "config": "params:pre_semantic_quality",
                },
                outputs=["pre_semantic_quality_report", "pre_semantic_quality_windows"],
                name="video_simple_quality_gate",
                tags=["quality", "video_simple_quality"],
            ),
            node(
                func=run_stereo_summary_quality_node,
                inputs="params:stereo_summary_quality",
                outputs="stereo_summary_quality_report",
                name="stereo_summary_quality",
                tags=["quality", "stereo"],
            ),
            node(
                func=run_stereo_segment_quality_node,
                inputs="params:stereo_segment_quality",
                outputs="stereo_segment_quality_report",
                name="stereo_segment_quality",
                tags=["quality", "stereo"],
            ),
            node(
                func=run_depth_quality_node_from_config,
                inputs="params:depth_quality",
                outputs="depth_quality_report",
                name="depth_quality",
                tags=["quality", "depth"],
            ),
            node(
                func=run_depth_numeric_quality_node,
                inputs="params:depth_numeric_quality",
                outputs="depth_numeric_quality_report",
                name="depth_numeric_quality",
                tags=["quality", "depth"],
            ),
        ]
    )
