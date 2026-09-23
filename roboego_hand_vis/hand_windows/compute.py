from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboego_hand_vis.hand_windows.logic import HandWindowConfig, write_hand_estimation_windows


@dataclass(frozen=True)
class HandWindowsStageConfig:
    stage1_actions: Path
    processing_plan: Path
    output_json: Path
    require_both_hands: bool
    hand_window_mode: str
    context_expand_s: int
    merge_gap_s: int
    merge_after_expand: bool
    min_window_s: int
    min_priority: str | None
    min_core_duration_s: int
    fps: int


def run_hand_windows_stage(config: HandWindowsStageConfig) -> dict[str, Any]:
    payload = write_hand_estimation_windows(
        stage1_actions_json=config.stage1_actions,
        processing_plan_json=config.processing_plan,
        output_json=config.output_json,
        config=HandWindowConfig(
            require_both_hands=config.require_both_hands,
            hand_window_mode=config.hand_window_mode,
            context_expand_s=config.context_expand_s,
            merge_gap_s=config.merge_gap_s,
            merge_after_expand=config.merge_after_expand,
            min_window_s=config.min_window_s,
            min_priority=config.min_priority,
            min_core_duration_s=config.min_core_duration_s,
            fps=config.fps,
        ),
    )
    return {
        "stage": "hand_windows",
        "status": "complete",
        "stage1_actions": str(config.stage1_actions),
        "processing_plan": str(config.processing_plan),
        "output_json": str(config.output_json),
        "window_count": int(payload.get("window_count", 0) or 0),
    }
