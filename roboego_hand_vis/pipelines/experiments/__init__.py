from __future__ import annotations

from roboego_hand_vis.pipelines._smoke import make_stage_pipeline


def create_pipeline(**kwargs):
    return make_stage_pipeline("experiments")
