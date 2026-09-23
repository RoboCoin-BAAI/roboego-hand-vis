from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboego_hand_vis.processing_plan.logic import ProcessingPlanConfig, write_processing_plan


@dataclass(frozen=True)
class ProcessingPlanStageConfig:
    stage1_actions: Path
    output_json: Path
    actions_json: Path | None
    min_hold_interruption_s: int
    min_walk_interruption_s: int
    min_idle_interruption_s: int
    context_expand_s: int
    min_process_segment_s: int
    merge_process_gap_s: int


def run_processing_plan_stage(config: ProcessingPlanStageConfig) -> dict[str, Any]:
    payload = write_processing_plan(
        stage1_json=config.stage1_actions,
        output_json=config.output_json,
        actions_json=config.actions_json,
        config=ProcessingPlanConfig(
            min_hold_interruption_s=config.min_hold_interruption_s,
            min_walk_interruption_s=config.min_walk_interruption_s,
            min_idle_interruption_s=config.min_idle_interruption_s,
            context_expand_s=config.context_expand_s,
            min_process_segment_s=config.min_process_segment_s,
            merge_process_gap_s=config.merge_process_gap_s,
        ),
    )
    return {
        "stage": "processing_plan",
        "status": "complete",
        "stage1_actions": str(config.stage1_actions),
        "actions_json": str(config.actions_json) if config.actions_json else "",
        "output_json": str(config.output_json),
        "segment_count": len(payload.get("segments", [])),
        "drop_segment_count": len(payload.get("drop_segments", [])),
    }
