from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


STATE_FILENAME = "pipeline_state.json"
DEFAULT_DASHBOARD_PATH = Path("reports/pipeline_dashboard/index.html")
STAGE_ORDER = [
    "raw_quality",
    "raw_quality_gate",
    "video_simple_quality",
    "video_simple_quality_gate",
    "qwen_annotation",
    "post_annotation_actions",
    "processing_plan",
    "hand_windows",
    "hand_visibility_gate",
    "materialize_auto",
    "prepare_stereo",
    "stereo_summary_quality",
    "stereo_quality",
    "fast_foundation_depth",
    "depth_numeric_quality",
    "archive_depth_cache",
    "hamer",
    "hand_filter",
    "joints21_pose_gate",
    "depth_anchored_joints21",
    "final_hand_result",
    "sam3_objects",
]
WORKFLOW_ALIAS_STAGES = {"base_prepare", "phase1_planning", "phase1_prepare_stereo", "phase1_depth"}
WORKFLOW_ALIAS_EXPANSIONS = {
    "base_prepare": [
        "raw_quality",
        "video_simple_quality",
        "qwen_annotation",
        "post_annotation_actions",
        "processing_plan",
        "hand_windows",
        "materialize_auto",
        "prepare_stereo",
    ],
    "phase1_depth": [
        "materialize_auto",
        "prepare_stereo",
        "stereo_summary_quality",
        "stereo_quality",
        "fast_foundation_depth",
        "depth_numeric_quality",
        "archive_depth_cache",
    ],
}
VALID_STATUSES = {"pending", "processing", "success", "fail", "skipped", "blocked"}
QUALITY_GATE_STAGES = {"raw_quality_gate", "video_simple_quality_gate"}


def read_pipeline_state(session: Path) -> dict[str, Any]:
    session = Path(session).expanduser().resolve()
    state_path = session / STATE_FILENAME
    if not state_path.is_file():
        return {"session": str(session), "stages": {}}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"session": str(session), "stages": {}}
    if not isinstance(payload, dict):
        return {"session": str(session), "stages": {}}
    payload.setdefault("session", str(session))
    payload.setdefault("stages", {})
    _expand_legacy_workflow_aliases(payload)
    return payload


def write_pipeline_state(session: Path, state: dict[str, Any]) -> Path:
    session = Path(session).expanduser().resolve()
    state["session"] = str(session)
    state.setdefault("stages", {})
    state_path = session / STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(state_path)
    return state_path


def update_stage_state(
    session: Path,
    stage: str,
    status: str,
    *,
    command: str | None = None,
    duration_s: float | None = None,
    outputs: list[str] | None = None,
    error: str | None = None,
    stdout_tail: str | None = None,
    stderr_tail: str | None = None,
    returncode: int | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid pipeline stage status {status!r}; expected one of {sorted(VALID_STATUSES)}")
    state = read_pipeline_state(session)
    stages = state.setdefault("stages", {})
    record = dict(stages.get(stage) or {})
    record["status"] = status
    record["updated_at"] = _now_iso()
    if status == "processing":
        record["started_at"] = record.get("started_at") or record["updated_at"]
        record["finished_at"] = None
    elif status in {"success", "fail", "blocked", "skipped"}:
        record["finished_at"] = record["updated_at"]
    if command is not None:
        record["command"] = command
    if duration_s is not None:
        record["duration_s"] = round(float(duration_s), 3)
    if outputs is not None:
        record["outputs"] = outputs
    if error is not None or status == "success":
        record["error"] = error
    if stdout_tail is not None:
        record["stdout_tail"] = stdout_tail
    if stderr_tail is not None:
        record["stderr_tail"] = stderr_tail
    if returncode is not None:
        record["returncode"] = int(returncode)
    stages[stage] = record
    write_pipeline_state(session, state)
    return state


def discover_sessions(root: Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if _looks_like_session(root):
        return [root]
    sessions: set[Path] = set()
    sessions.update(_discover_baai_divide_sessions(root))
    if not sessions:
        sessions.update(_discover_sessions_pruned(root))
    return sorted(sessions)


def _discover_baai_divide_sessions(root: Path) -> set[Path]:
    sessions: set[Path] = set()
    roots = sorted(root.glob("cp_*")) if root.name != "divide" else []
    if root.name.startswith("cp_"):
        roots = [root]
    for cp_dir in roots:
        if not cp_dir.is_dir():
            continue
        for parent_session in sorted(cp_dir.glob("session_*")):
            divide = parent_session / "divide"
            if not divide.is_dir():
                continue
            for session in sorted(divide.glob("session_*")):
                if _looks_like_session(session):
                    sessions.add(session.resolve())
    if root.name == "divide":
        for session in sorted(root.glob("session_*")):
            if _looks_like_session(session):
                sessions.add(session.resolve())
    return sessions


def _discover_sessions_pruned(root: Path) -> set[Path]:
    sessions: set[Path] = set()
    skip_dirs = {
        ".git",
        "__pycache__",
        "annotation",
        "dataset_manifest",
        "depth",
        "hand",
        "hand_windows",
        "objects",
        "processed",
        "quality",
        "rectified",
        "segments",
    }
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        if current_path != root and current_path.name in skip_dirs:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in skip_dirs]
        if "meta.json" not in filenames:
            continue
        if _looks_like_session(current_path):
            sessions.add(current_path.resolve())
            dirnames[:] = []
    return sessions


def run_batch_stage(
    *,
    root: Path,
    dataset: str,
    stage: str,
    workers: int = 1,
    config: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    dashboard: bool = False,
    dashboard_path: Path | None = None,
) -> dict[str, Any]:
    sessions = discover_sessions(root)
    if not sessions:
        raise FileNotFoundError(f"No sessions found under {Path(root).expanduser().resolve()}")

    results: list[dict[str, Any]] = []
    if workers <= 1:
        for session in sessions:
            results.append(
                run_one_stage(
                    session=session,
                    dataset=dataset,
                    stage=stage,
                    config=config,
                    force=force,
                    dry_run=dry_run,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    run_one_stage,
                    session=session,
                    dataset=dataset,
                    stage=stage,
                    config=config,
                    force=force,
                    dry_run=dry_run,
                )
                for session in sessions
            ]
            for future in as_completed(futures):
                results.append(future.result())

    status_counts = dict(Counter(str(result.get("status", "unknown")) for result in results))
    summary = {
        "root": str(Path(root).expanduser().resolve()),
        "dataset": dataset,
        "stage": stage,
        "session_count": len(sessions),
        "status_counts": status_counts,
        "results": sorted(results, key=lambda item: item.get("session", "")),
    }
    if dashboard:
        summary["dashboard"] = str(render_dashboard(sessions, dashboard_path or DEFAULT_DASHBOARD_PATH))
    return summary


def run_one_stage(
    *,
    session: Path,
    dataset: str,
    stage: str,
    config: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    state = read_pipeline_state(session)
    existing = state.get("stages", {}).get(stage, {})
    if not force and existing.get("status") == "success":
        return {"session": str(Path(session).resolve()), "stage": stage, "status": "skipped", "reason": "already_success"}

    argv = _stage_command(session=session, dataset=dataset, stage=stage, config=config, force=force)
    command = " ".join(_quote(part) for part in argv)
    if dry_run:
        print(command)
        return {"session": str(Path(session).resolve()), "stage": stage, "status": "skipped", "reason": "dry_run"}

    update_stage_state(session, stage, "processing", command=command)
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    duration_s = time.monotonic() - started
    stdout_tail = _tail(completed.stdout)
    stderr_tail = _tail(completed.stderr)
    status = "success" if completed.returncode == 0 else "blocked" if completed.returncode == 2 else "fail"
    returncode = completed.returncode
    error = None if status == "success" else (stderr_tail or stdout_tail or f"exit code {completed.returncode}")
    if completed.returncode == 0:
        artifact_outcome = stage_artifact_outcome(session, stage)
        if artifact_outcome is not None:
            status = str(artifact_outcome["status"])
            returncode = int(artifact_outcome.get("returncode", returncode))
            error = artifact_outcome.get("error")
    update_stage_state(
        session,
        stage,
        status,
        command=command,
        duration_s=duration_s,
        error=error,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        returncode=returncode,
    )
    print(f"[{status}] {session} stage={stage} duration_s={duration_s:.1f}")
    return {
        "session": str(Path(session).resolve()),
        "stage": stage,
        "status": status,
        "duration_s": round(duration_s, 3),
        "returncode": returncode,
        "error": error,
    }


def stage_artifact_outcome(session: Path, stage: str) -> dict[str, Any] | None:
    session = Path(session).expanduser().resolve()
    if stage == "raw_quality":
        return _raw_quality_artifact_outcome(session / "annotation" / "quality_report.json")
    if stage == "video_simple_quality":
        return _generic_passed_report_outcome(
            stage=stage,
            report_path=session / "annotation" / "video_simple_quality_left.json",
            failed_key="failed_intervals",
        )
    if stage == "hand_visibility_gate":
        return _generic_passed_report_outcome(
            stage=stage,
            report_path=session / "annotation" / "hand_visibility_quality_gate_left_v1.json",
            failed_key="blocked_window_ids",
        )
    if stage == "fast_foundation_depth":
        return _fast_foundation_depth_artifact_outcome(session)
    return None


def _raw_quality_artifact_outcome(report_path: Path) -> dict[str, Any] | None:
    report = _read_json_report(report_path)
    if not isinstance(report, dict):
        return None
    gate = report.get("raw_headband_quality_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not False:
        return None
    failed_checks = gate.get("failed_checks", [])
    lines = [
        "RAW QUALITY GATE FAILED",
        f"failed_checks={failed_checks}",
        f"report={report_path}",
    ]
    for check in gate.get("checks", []):
        if not isinstance(check, dict) or check.get("passed") is not False:
            continue
        detail = f"- {check.get('name')}"
        if check.get("stream"):
            detail += f":{check.get('stream')}"
        if check.get("message"):
            detail += f" | {check.get('message')}"
        if "value" in check:
            detail += f" | value={check.get('value')}"
        if "threshold" in check:
            detail += f" | threshold={check.get('threshold')}"
        lines.append(detail)
    return {"status": "blocked", "returncode": 2, "error": "\n".join(lines)}


def _generic_passed_report_outcome(*, stage: str, report_path: Path, failed_key: str) -> dict[str, Any] | None:
    report = _read_json_report(report_path)
    if not isinstance(report, dict) or report.get("passed") is not False:
        return None
    return {
        "status": "blocked",
        "returncode": 2,
        "error": (
            f"{stage} gate failed\n"
            f"{failed_key}={report.get(failed_key, [])}\n"
            f"report={report_path}"
        ),
    }


def _read_json_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _fast_foundation_depth_artifact_outcome(session: Path) -> dict[str, Any] | None:
    units = _find_prepared_units(session)
    if not units:
        return {
            "status": "blocked",
            "returncode": 2,
            "error": (
                "FAST FOUNDATION DEPTH OUTPUT MISSING\n"
                f"reason=no_prepared_dataset_manifest\n"
                f"session={session}\n"
                "run prepare_stereo or base_prepare first"
            ),
        }
    missing: list[str] = []
    for unit in units:
        depth_dir = unit / "depth" / "fast_foundation"
        meta = depth_dir / "fast_foundation_stereo_video_meta.json"
        png_dir = depth_dir / "depth_uint16_png"
        has_png = png_dir.is_dir() and any(png_dir.glob("*_depth_mm.png"))
        if not meta.is_file() and not has_png:
            missing.append(str(depth_dir))
    if not missing:
        return None
    stereo_reports = [str(unit / "quality/stereo_summary/stereo_quality_report.json") for unit in units]
    return {
        "status": "blocked",
        "returncode": 2,
        "error": (
            "FAST FOUNDATION DEPTH OUTPUT MISSING\n"
            f"missing_depth_outputs={missing}\n"
            f"required_before_depth=stereo_summary_quality\n"
            f"stereo_quality_reports={stereo_reports}\n"
            "If the depth node finished in 0.0 sec, it was probably blocked by a missing or unusable stereo summary report."
        ),
    }


def _find_prepared_units(session: Path) -> list[Path]:
    search_specs = [
        (session / "hand_windows", "handwin_*/dataset_manifest/dataset_manifest.json"),
        (session / "segments", "*/hand_windows/handwin_*/dataset_manifest/dataset_manifest.json"),
        (session / "segments", "seg_*/dataset_manifest/dataset_manifest.json"),
        (session, "dataset_manifest/dataset_manifest.json"),
    ]
    for root, pattern in search_specs:
        if not root.exists():
            continue
        units = sorted({path.parents[1].resolve() for path in root.glob(pattern) if path.is_file()})
        if units:
            return units
    return []


def render_dashboard(sessions: list[Path], output_html: Path) -> Path:
    output_html = Path(output_html).expanduser()
    if not output_html.is_absolute():
        output_html = Path.cwd() / output_html
    output_html.parent.mkdir(parents=True, exist_ok=True)
    ordered_sessions = sorted(Path(item).resolve() for item in sessions)
    stages = _dashboard_stages(ordered_sessions)
    rows = [_session_row(Path(session), stages) for session in ordered_sessions]
    counts = Counter()
    for session in sessions:
        state = read_pipeline_state(session)
        session_stages = state.get("stages", {})
        for stage in stages:
            if stage in WORKFLOW_ALIAS_STAGES:
                continue
            record = _dashboard_stage_record(Path(session), session_stages, stage)
            status = str(record.get("status", "pending"))
            if status != "pending":
                counts[status] += 1
    output_html.write_text(_dashboard_html(rows, counts, stages), encoding="utf-8")
    return output_html


def print_state_report(root: Path) -> dict[str, Any]:
    sessions = discover_sessions(root)
    stage_counts: dict[str, Counter] = {}
    for session in sessions:
        state = read_pipeline_state(session)
        session_stages = state.get("stages", {})
        for stage in _summary_stages(session, session_stages):
            if stage in WORKFLOW_ALIAS_STAGES:
                continue
            record = _dashboard_stage_record(session, session_stages, stage)
            status = str(record.get("status", "pending"))
            if status == "pending":
                continue
            stage_counts.setdefault(stage, Counter())[status] += 1
    payload = {
        "root": str(Path(root).expanduser().resolve()),
        "session_count": len(sessions),
        "stage_counts": {stage: dict(counts) for stage, counts in sorted(stage_counts.items())},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def batch_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one RoboEgo pipeline stage across many sessions.")
    parser.add_argument("--root", required=True, help="Root directory containing session folders.")
    parser.add_argument("--dataset", required=True, help="Dataset config name, for example raw.")
    parser.add_argument("--stage", required=True, help="Stage or workflow name from conf/datasets/<dataset>.yml.")
    parser.add_argument("--config", default="", help="Optional explicit dataset yaml path.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Re-run sessions whose stage already succeeded.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dashboard", action="store_true", help="Generate an HTML dashboard after this run.")
    parser.add_argument("--dashboard-path", default=str(DEFAULT_DASHBOARD_PATH))
    args = parser.parse_args(argv)

    summary = run_batch_stage(
        root=Path(args.root),
        dataset=args.dataset,
        stage=args.stage,
        workers=max(1, args.workers),
        config=Path(args.config).expanduser() if args.config else None,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        dashboard=bool(args.dashboard),
        dashboard_path=Path(args.dashboard_path),
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0


def report_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize RoboEgo per-session pipeline_state.json files.")
    parser.add_argument("--root", required=True, help="Root directory containing session folders.")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--dashboard-path", default=str(DEFAULT_DASHBOARD_PATH))
    args = parser.parse_args(argv)
    sessions = discover_sessions(Path(args.root))
    print_state_report(Path(args.root))
    if args.dashboard:
        output = render_dashboard(sessions, Path(args.dashboard_path))
        print(f"Wrote dashboard: {output}")
    return 0


def _looks_like_session(path: Path) -> bool:
    if any(part in {"segments", "hand_windows", "processed", "rectified"} for part in path.parts):
        return False
    return (path / "meta.json").is_file() and (path / "videos").is_dir()


def _expand_legacy_workflow_aliases(state: dict[str, Any]) -> None:
    stages = state.setdefault("stages", {})
    if not isinstance(stages, dict):
        state["stages"] = {}
        return
    for alias, expanded_stages in WORKFLOW_ALIAS_EXPANSIONS.items():
        alias_record = stages.get(alias)
        if not isinstance(alias_record, dict):
            continue
        for stage in expanded_stages:
            if stage in stages:
                continue
            record = dict(alias_record)
            record["derived_from"] = alias
            stages[stage] = record


def _stage_command(*, session: Path, dataset: str, stage: str, config: Path | None, force: bool = False) -> list[str]:
    argv = [
        sys.executable,
        "scripts/run_pipeline_stage.py",
        "--dataset",
        dataset,
        "--session",
        str(Path(session).resolve()),
        "--stage",
        stage,
    ]
    if config is not None:
        argv.extend(["--config", str(Path(config).expanduser())])
    if force:
        argv.append("--force")
    return argv


def _dashboard_stages(sessions: list[Path]) -> list[str]:
    stages = list(STAGE_ORDER)
    for session in sessions:
        state = read_pipeline_state(session)
        for stage in sorted(state.get("stages", {})):
            if stage in WORKFLOW_ALIAS_STAGES:
                continue
            if stage not in stages:
                stages.append(stage)
    return stages


def _summary_stages(session: Path, stages: dict[str, Any]) -> list[str]:
    names = [stage for stage in stages if stage not in WORKFLOW_ALIAS_STAGES]
    for stage in QUALITY_GATE_STAGES:
        record = _quality_gate_dashboard_record(session, stage)
        if record.get("status") != "pending" and stage not in names:
            names.append(stage)
    return sorted(names, key=_stage_sort_key)


def _stage_sort_key(stage: str) -> tuple[int, str]:
    try:
        return (STAGE_ORDER.index(stage), stage)
    except ValueError:
        return (len(STAGE_ORDER), stage)


def _session_row(session: Path, stages_to_show: list[str]) -> str:
    state = read_pipeline_state(session)
    stages = state.get("stages", {})
    cells = []
    for stage in stages_to_show:
        record = _dashboard_stage_record(session, stages, stage)
        status = str(record.get("status", "pending"))
        title = _stage_title(stage, record)
        cells.append(f'<td class="status {html.escape(status)}" title="{title}">{html.escape(status)}</td>')
    failures = [
        f"{stage}: {record.get('error')}"
        for stage, record in stages.items()
        if stage not in WORKFLOW_ALIAS_STAGES
        and record.get("status") in {"fail", "blocked"}
        and record.get("error")
    ]
    failure_summary = _tail("\n".join(failures), lines=6)
    return (
        "<tr>"
        f"<th>{html.escape(session.name)}</th>"
        f"<td class=\"path\">{html.escape(str(session))}</td>"
        + "".join(cells)
        + f"<td class=\"error\">{html.escape(failure_summary)}</td>"
        "</tr>"
    )


def _dashboard_stage_record(session: Path, stages: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage in QUALITY_GATE_STAGES:
        return _quality_gate_dashboard_record(session, stage)
    return stages.get(stage) or {"status": "pending"}


def _quality_gate_dashboard_record(session: Path, stage: str) -> dict[str, Any]:
    if stage == "raw_quality_gate":
        outcome = stage_artifact_outcome(session, "raw_quality")
        if outcome is not None:
            return outcome
        report = _read_json_report(Path(session) / "annotation" / "quality_report.json")
        gate = report.get("raw_headband_quality_gate") if isinstance(report, dict) else None
        if isinstance(gate, dict) and gate.get("passed") is True:
            return {"status": "success", "report": str(Path(session) / "annotation" / "quality_report.json")}
        return {"status": "pending"}
    if stage == "video_simple_quality_gate":
        outcome = stage_artifact_outcome(session, "video_simple_quality")
        if outcome is not None:
            return outcome
        report_path = Path(session) / "annotation" / "video_simple_quality_left.json"
        report = _read_json_report(report_path)
        if isinstance(report, dict) and report.get("passed") is True:
            return {"status": "success", "report": str(report_path)}
        return {"status": "pending"}
    return {"status": "pending"}


def _stage_title(stage: str, record: dict[str, Any]) -> str:
    parts = [stage]
    if record.get("duration_s") is not None:
        parts.append(f"duration_s={record['duration_s']}")
    if record.get("derived_from"):
        parts.append(f"derived_from={record['derived_from']}")
    if record.get("error"):
        parts.append(str(record["error"]))
    if record.get("report"):
        parts.append(f"report={record['report']}")
    return html.escape(" | ".join(parts), quote=True)


def _dashboard_html(rows: list[str], counts: Counter, stages: list[str]) -> str:
    header_cells = "".join(f"<th>{html.escape(stage)}</th>" for stage in stages)
    count_text = " ".join(f"{html.escape(status)}={count}" for status, count in sorted(counts.items()))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Pipeline Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f6f8fb; }}
    header {{ padding: 18px 24px; background: #ffffff; border-bottom: 1px solid #dde3ea; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta {{ color: #52616f; font-size: 13px; }}
    main {{ padding: 18px 24px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: max-content; min-width: 100%; background: #ffffff; }}
    th, td {{ border: 1px solid #d9e1ea; padding: 7px 8px; font-size: 12px; vertical-align: top; }}
    th {{ background: #eef3f8; position: sticky; top: 0; z-index: 1; }}
    th:first-child {{ min-width: 220px; text-align: left; }}
    .path {{ min-width: 420px; color: #52616f; }}
    .error {{ min-width: 360px; max-width: 520px; white-space: pre-wrap; color: #7c2d12; }}
    .status {{ text-align: center; min-width: 76px; font-weight: 700; }}
    .success {{ background: #dcfce7; color: #166534; }}
    .fail {{ background: #fee2e2; color: #991b1b; }}
    .blocked {{ background: #ffedd5; color: #9a3412; }}
    .processing {{ background: #dbeafe; color: #1d4ed8; }}
    .skipped {{ background: #f3f4f6; color: #4b5563; }}
    .pending {{ background: #ffffff; color: #9ca3af; }}
  </style>
</head>
<body>
  <header>
    <h1>Pipeline Dashboard</h1>
    <div class="meta">Generated at {_now_iso()} | {count_text}</div>
  </header>
  <main>
    <table>
      <thead><tr><th>session</th><th>path</th>{header_cells}<th>failure summary</th></tr></thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def _tail(text: str, *, lines: int = 80) -> str:
    if not text:
        return ""
    split = text.strip().splitlines()
    return "\n".join(split[-lines:])


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _quote(value: str) -> str:
    if not value:
        return "''"
    if any(char.isspace() for char in value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value
