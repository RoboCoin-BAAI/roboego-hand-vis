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
    # Privacy consumes completion reports only in the default graph, ensuring
    # it starts after the processing stages even with a parallel Kedro runner.
    processing_stages = ("quality", "semantic", "datasets", "restoration", "depth", "egohand", "objects")
    completion_reports = sorted(set().union(*(pipelines[name].outputs() for name in processing_stages)))
    default_privacy = privacy.create_pipeline(after=completion_reports)
    pipelines["__default__"] = sum(
        (pipeline for name, pipeline in pipelines.items() if name != "privacy"),
        default_privacy,
    )
    return pipelines
