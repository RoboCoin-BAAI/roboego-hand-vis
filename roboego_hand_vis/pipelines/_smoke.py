from __future__ import annotations

from kedro.pipeline import Pipeline, node


def make_stage_pipeline(stage: str) -> Pipeline:
    def _smoke_node() -> dict[str, str]:
        return {"stage": stage, "status": "skipped", "reason": "not_configured"}

    _smoke_node.__name__ = f"{stage}_smoke_node"
    return Pipeline(
        [
            node(
                func=_smoke_node,
                inputs=None,
                outputs=f"{stage}_smoke_report",
                name=f"{stage}_smoke",
                tags=[stage],
            )
        ]
    )
