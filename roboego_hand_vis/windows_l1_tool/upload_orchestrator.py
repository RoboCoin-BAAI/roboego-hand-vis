from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

TOOL_DIR = Path(__file__).resolve().parent

try:
    from roboego_hand_vis.windows_l1_tool.upload_core import MANIFEST_FILENAME, load_uploaded_folders, upload_folder
except ModuleNotFoundError:
    if str(TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(TOOL_DIR))
    from upload_core import MANIFEST_FILENAME, load_uploaded_folders, upload_folder  # noqa: E402


DEFAULT_SKIPPED_UPLOAD_NAME = "raw_headband_quality_skipped_uploads.json"


@dataclass(frozen=True)
class QualityFilterResult:
    passed: list[Path]
    skipped: list[dict]


def _quality_report_path(session: Path) -> Path:
    return session / "annotation" / "quality_report.json"


def _load_gate(session: Path) -> tuple[dict | None, str | None]:
    report_path = _quality_report_path(session)
    if not report_path.exists():
        return None, "quality_report_missing"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "quality_report_invalid_json"
    gate = report.get("raw_headband_quality_gate")
    if not isinstance(gate, dict):
        return None, "quality_gate_missing"
    return gate, None


def filter_quality_passed_sessions(sessions: Iterable[str | Path]) -> QualityFilterResult:
    passed: list[Path] = []
    skipped: list[dict] = []
    for raw_session in sessions:
        session = Path(raw_session).resolve()
        gate, reason = _load_gate(session)
        if reason is not None:
            skipped.append({"local_dir": str(session), "reason": reason, "report_path": str(_quality_report_path(session))})
            continue
        if bool(gate.get("passed")):
            passed.append(session)
            continue
        skipped.append(
            {
                "local_dir": str(session),
                "reason": "quality_gate_failed",
                "report_path": str(_quality_report_path(session)),
                "failed_checks": list(gate.get("failed_checks") or []),
                "warnings": list(gate.get("warnings") or []),
            }
        )
    return QualityFilterResult(passed=passed, skipped=skipped)


def upload_quality_passed_sessions(
    sessions: Iterable[str | Path],
    *,
    args,
    skipped_json: str | Path | None = None,
) -> dict:
    filter_result = filter_quality_passed_sessions(sessions)
    manifest_path = Path(args.manifest) if getattr(args, "manifest", None) else Path(MANIFEST_FILENAME)
    uploaded_set = load_uploaded_folders(manifest_path)
    failed_uploads: list[str] = []
    uploaded_count = 0
    skipped_sessions = list(filter_result.skipped)

    for session in filter_result.passed:
        if str(session) in uploaded_set:
            skipped_sessions.append({"local_dir": str(session), "reason": "already_uploaded"})
            continue
        ok = upload_folder(
            local_dir=session,
            ak=args.ak,
            sk=args.sk,
            project_id=args.project_id,
            manifest_path=manifest_path,
            dry_run=bool(getattr(args, "dry_run", False)),
            task_id=getattr(args, "task_id", None),
        )
        if ok:
            uploaded_count += 1
        else:
            failed_uploads.append(str(session))

    skipped_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "skipped_sessions": skipped_sessions,
    }
    skipped_path = Path(skipped_json) if skipped_json is not None else Path(DEFAULT_SKIPPED_UPLOAD_NAME)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.write_text(json.dumps(skipped_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "total": len(filter_result.passed) + len(filter_result.skipped),
        "uploaded": uploaded_count,
        "upload_failed": len(failed_uploads),
        "skipped": len(skipped_sessions),
        "failed_uploads": failed_uploads,
        "skipped_json": str(skipped_path.resolve()),
    }
