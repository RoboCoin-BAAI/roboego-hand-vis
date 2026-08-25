"""Semantic annotation pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from roboego_hand_vis.pipelines.semantic.nodes import (
    run_build_hand_estimation_windows_node,
    run_build_processing_plan_node,
    run_qwen_frame_annotation_node,
    run_stage1_action_extraction_chunks_node,
    run_stage1_action_text_resolver_chunks_node,
    run_stage1_action_text_review_chunks_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_qwen_frame_annotation_node,
                inputs={
                    "pre_semantic_quality_windows": "pre_semantic_quality_windows",
                    "config": "params:qwen_frame_annotation",
                },
                outputs="qwen_frame_annotation_report",
                name="qwen_frame_annotation",
                tags=["semantic", "qwen"],
            ),
            node(
                func=run_stage1_action_extraction_chunks_node,
                inputs={"config": "params:stage1_action_chunks"},
                outputs="stage1_action_chunks_report",
                name="stage1_action_extraction_chunks",
                tags=["semantic", "stage1", "chunks"],
            ),
            node(
                func=run_stage1_action_text_review_chunks_node,
                inputs={
                    "stage1_action_chunks_report": "stage1_action_chunks_report",
                    "config": "params:stage1_action_chunks",
                },
                outputs="stage1_action_text_review_chunks_report",
                name="stage1_action_text_review_chunks",
                tags=["semantic", "stage1", "chunks"],
            ),
            node(
                func=run_stage1_action_text_resolver_chunks_node,
                inputs={
                    "stage1_action_text_review_chunks_report": "stage1_action_text_review_chunks_report",
                    "config": "params:stage1_action_chunks",
                },
                outputs="stage1_action_text_resolver_chunks_report",
                name="stage1_action_text_resolver_chunks",
                tags=["semantic", "stage1", "chunks", "qwen_text"],
            ),
            node(
                func=run_build_processing_plan_node,
                inputs={"config": "params:processing_plan"},
                outputs="processing_plan_report",
                name="build_processing_plan",
                tags=["semantic", "processing_plan"],
            ),
            node(
                func=run_build_hand_estimation_windows_node,
                inputs={"config": "params:hand_estimation_windows"},
                outputs="hand_estimation_windows_report",
                name="build_hand_estimation_windows",
                tags=["semantic", "hand_windows"],
            ),
        ]
    )
