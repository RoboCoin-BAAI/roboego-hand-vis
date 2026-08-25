"""Ego-hand pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from roboego_hand_vis.pipelines.egohand.nodes import (
    run_depth_anchored_joints21_node,
    run_final_hand_result_node,
    run_hamer_windows_node,
    run_joints21_pose_gate_node,
    run_legacy_export_joints21_node,
    run_legacy_mesh_depth_alignment_node,
    run_legacy_track_fill_smooth_node,
    run_refined_temporal_gate_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_hamer_windows_node,
                inputs="params:hamer_windows",
                outputs="hamer_windows_report",
                name="hamer_windows",
                tags=["egohand", "hamer"],
            ),
            node(
                func=run_refined_temporal_gate_node,
                inputs="params:refined_temporal_gate",
                outputs="refined_temporal_gate_report",
                name="refined_temporal_gate",
                tags=["egohand", "temporal_gate"],
            ),
            node(
                func=run_final_hand_result_node,
                inputs="params:final_hand_result",
                outputs="final_hand_result_report",
                name="final_hand_result",
                tags=["egohand", "final_hand_result"],
            ),
            node(
                func=run_joints21_pose_gate_node,
                inputs="params:joints21_pose_gate",
                outputs="joints21_pose_gate_report",
                name="joints21_pose_gate",
                tags=["egohand", "joints21_pose_gate"],
            ),
            node(
                func=run_depth_anchored_joints21_node,
                inputs="params:depth_anchored_joints21",
                outputs="depth_anchored_joints21_report",
                name="depth_anchored_joints21",
                tags=["egohand", "depth_anchored_joints21"],
            ),
            node(
                func=run_legacy_mesh_depth_alignment_node,
                inputs="params:legacy_mesh_depth_alignment",
                outputs="legacy_mesh_depth_alignment_report",
                name="legacy_mesh_depth_alignment",
                tags=["egohand", "legacy_depth_alignment"],
            ),
            node(
                func=run_legacy_track_fill_smooth_node,
                inputs="params:legacy_track_fill_smooth",
                outputs="legacy_track_fill_smooth_report",
                name="legacy_track_fill_smooth",
                tags=["egohand", "legacy_depth_alignment"],
            ),
            node(
                func=run_legacy_export_joints21_node,
                inputs="params:legacy_export_joints21",
                outputs="legacy_export_joints21_report",
                name="legacy_export_joints21",
                tags=["egohand", "legacy_depth_alignment"],
            ),
        ]
    )
