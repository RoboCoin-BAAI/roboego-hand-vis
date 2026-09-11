"""Privacy pipeline definition."""

from __future__ import annotations

from collections.abc import Sequence

from kedro.pipeline import Pipeline, node

from roboego_hand_vis.pipelines.privacy.nodes import (
    run_egoblur_temporal_face_blur_node,
    run_video_privacy_blur_node,
)


def _run_after_processing(config, **stage_reports):
    """Use upstream reports as scheduling dependencies without altering their data."""
    return run_video_privacy_blur_node(config)


def create_pipeline(*, after: Sequence[str] = (), **kwargs) -> Pipeline:
    # Only the default pipeline supplies these dependencies. The named privacy
    # pipeline remains runnable on an existing session without upstream stages.
    blur_inputs = {"config": "params:video_privacy_blur"}
    blur_inputs.update({f"stage_report_{i}": report for i, report in enumerate(after)})
    return Pipeline(
        [
            node(
                func=_run_after_processing if after else run_video_privacy_blur_node,
                inputs=blur_inputs if after else "params:video_privacy_blur",
                outputs="video_privacy_blur_report",
                name="video_privacy_blur",
                tags=["privacy", "video_privacy_blur"],
            ),
            node(
                func=run_egoblur_temporal_face_blur_node,
                inputs="params:egoblur_temporal_face_blur",
                outputs="egoblur_temporal_face_blur_report",
                name="egoblur_temporal_face_blur",
                tags=["privacy", "egoblur"],
            ),
        ]
    )
