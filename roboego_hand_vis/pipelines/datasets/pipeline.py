"""Dataset preparation pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from roboego_hand_vis.pipelines.datasets.nodes import (
    run_export_direct_hand_window_segments_node,
    run_export_hand_window_segments_node,
    run_export_selected_video_segments_node,
    run_prepare_selected_segments_node,
    run_prepare_stereo_unit_node,
    run_resolve_dataset_manifest_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_resolve_dataset_manifest_node,
                inputs="params:resolve_dataset_manifest",
                outputs="resolved_dataset_manifest_report",
                name="resolve_dataset_manifest",
                tags=["datasets", "manifest"],
            ),
            node(
                func=run_export_selected_video_segments_node,
                inputs="params:export_selected_video_segments",
                outputs="export_selected_video_segments_report",
                name="export_selected_video_segments",
                tags=["datasets", "segments"],
            ),
            node(
                func=run_prepare_selected_segments_node,
                inputs={
                    "export_selected_video_segments_report": "export_selected_video_segments_report",
                    "config": "params:prepare_selected_segments",
                },
                outputs="prepare_selected_segments_report",
                name="prepare_selected_segments",
                tags=["datasets", "segments", "rectify"],
            ),
            node(
                func=run_export_hand_window_segments_node,
                inputs={
                    "prepare_selected_segments_report": "prepare_selected_segments_report",
                    "config": "params:export_hand_window_segments",
                },
                outputs="export_hand_window_segments_report",
                name="export_hand_window_segments",
                tags=["datasets", "segments", "hand_windows"],
            ),
            node(
                func=run_export_direct_hand_window_segments_node,
                inputs="params:export_direct_hand_window_segments",
                outputs="export_direct_hand_window_segments_report",
                name="export_direct_hand_window_segments",
                tags=["datasets", "hand_windows", "direct"],
            ),
            node(
                func=run_prepare_stereo_unit_node,
                inputs="params:prepare_stereo_unit",
                outputs="prepare_stereo_unit_report",
                name="prepare_stereo_unit",
                tags=["datasets", "stereo", "rectify"],
            ),
        ]
    )
