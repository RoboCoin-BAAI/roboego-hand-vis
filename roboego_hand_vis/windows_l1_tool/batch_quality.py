from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

TOOL_DIR = Path(__file__).resolve().parent

try:
    from roboego_hand_vis.windows_l1_tool import local_raw_quality
    from roboego_hand_vis.windows_l1_tool.local_raw_quality import DEFAULT_CAMERA_STREAMS, DEFAULT_VIDEO_EXT
    from roboego_hand_vis.windows_l1_tool.local_raw_rules import evaluate_quality_report
    from roboego_hand_vis.windows_l1_tool.upload_core import find_all_cp_cs_folders
except ModuleNotFoundError:
    if str(TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(TOOL_DIR))
    import local_raw_quality  # noqa: E402
    from local_raw_quality import DEFAULT_CAMERA_STREAMS, DEFAULT_VIDEO_EXT  # noqa: E402
    from local_raw_rules import evaluate_quality_report  # noqa: E402
    from upload_core import find_all_cp_cs_folders  # noqa: E402


DEFAULT_FAILED_REPORT_NAME = "l1_quality_failed_sessions.json"
REPAIR_TIMESTAMP_FILES = True
MAX_REPAIR_DUPLICATE_RATE = 1.0
MAX_REPAIR_CONSECUTIVE_DUPLICATE_FRAMES = 5


def discover_cp_cs_sessions(roots: Iterable[str | Path]) -> list[Path]:
    sessions: list[Path] = []
    for root in roots:
        sessions.extend(find_all_cp_cs_folders(Path(root)))
    return sorted(dict.fromkeys(path.resolve() for path in sessions))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_quality_for_session(session: Path, *, overwrite: bool = True) -> dict:
    session = session.resolve()
    report_path = local_raw_quality.export_quality_report(
        session_dir=session,
        output_root=session,
        camera_streams=DEFAULT_CAMERA_STREAMS,
        start=0,
        num_frames=None,
        timestamp_tolerance_s=None,
        video_ext=DEFAULT_VIDEO_EXT,
        overwrite=overwrite,
        environment_name="windows_l1_tool",
        repair_timestamp_files=REPAIR_TIMESTAMP_FILES,
        max_repair_duplicate_rate=MAX_REPAIR_DUPLICATE_RATE,
        max_repair_consecutive_duplicate_frames=MAX_REPAIR_CONSECUTIVE_DUPLICATE_FRAMES,
        video_inspector=local_raw_quality.inspect_video,
    )
    report = _read_json(report_path)
    gate_result = evaluate_quality_report(report)
    report["raw_headband_quality_gate"] = gate_result
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "local_dir": str(session),
        "report_path": str(report_path),
        "passed": bool(gate_result.get("passed")),
        "failed_checks": list(gate_result.get("failed_checks") or []),
        "warnings": list(gate_result.get("warnings") or []),
    }


def batch_quality_check(
    roots: Iterable[str | Path],
    *,
    output_json: str | Path | None = None,
    overwrite: bool = True,
) -> dict:
    sessions = discover_cp_cs_sessions(roots)
    failed_sessions: list[dict] = []
    passed_count = 0
    error_count = 0

    for session in sessions:
        try:
            result = run_quality_for_session(session, overwrite=overwrite)
        except Exception as exc:
            error_count += 1
            failed_sessions.append(
                {
                    "local_dir": str(session),
                    "report_path": None,
                    "passed": False,
                    "failed_checks": ["quality_exception"],
                    "warnings": [],
                    "error": str(exc),
                }
            )
            continue
        if result["passed"]:
            passed_count += 1
        else:
            result = dict(result)
            result.setdefault("local_dir", str(session))
            result.pop("session", None)
            failed_sessions.append(result)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(sessions),
        "passed": passed_count,
        "failed": len(failed_sessions),
        "errors": error_count,
        "failed_sessions": failed_sessions,
    }
    output_path = Path(output_json) if output_json is not None else Path(DEFAULT_FAILED_REPORT_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["output_json"] = str(output_path.resolve())
    return summary
