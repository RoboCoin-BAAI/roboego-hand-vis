"""Kedro pipeline registry for RoboEgo processing stages."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from roboego_hand_vis.pipelines import (
    datasets,
    depth,
    egohand,
    experiments,
    export,
    objects,
    privacy,
    quality,
    restoration,
    semantic,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Return the named processing stages available to Kedro."""
    pipelines = {
        "datasets": datasets.create_pipeline(),
        "quality": quality.create_pipeline(),
        "restoration": restoration.create_pipeline(),
        "semantic": semantic.create_pipeline(),
        "depth": depth.create_pipeline(),
        "egohand": egohand.create_pipeline(),
        "objects": objects.create_pipeline(),
        "privacy": privacy.create_pipeline(),
        "export": export.create_pipeline(),
        "experiments": experiments.create_pipeline(),
    }
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
