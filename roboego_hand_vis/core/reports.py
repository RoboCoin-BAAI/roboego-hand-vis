"""Stage report utilities for pipeline observability."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_stage_report(
    path: str | Path,
    *,
    stage: str,
    status: str,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Write a compact JSON report for one processing stage."""
    report = {
        "stage": stage,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "params": params or {},
        "message": message,
    }
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
