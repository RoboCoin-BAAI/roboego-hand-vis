from __future__ import annotations

import argparse
import html
import json
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
    "video_simple_quality",
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
WORKFLOW_ALIAS_STAGES = {"base_prepare", "phase1_depth"}
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
    for meta_path in root.rglob("meta.json"):
        session = meta_path.parent
        if _looks_like_session(session):
            sessions.add(session.resolve())
    return sorted(sessions)


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

    argv = _stage_command(session=session, dataset=dataset, stage=stage, config=config)
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
    error = None if status == "success" else (stderr_tail or stdout_tail or f"exit code {completed.returncode}")
    update_stage_state(
        session,
        stage,
        status,
        command=command,
        duration_s=duration_s,
        error=error,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        returncode=completed.returncode,
    )
    print(f"[{status}] {session} stage={stage} duration_s={duration_s:.1f}")
    return {
        "session": str(Path(session).resolve()),
        "stage": stage,
        "status": status,
        "duration_s": round(duration_s, 3),
        "returncode": completed.returncode,
        "error": error,
    }


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
        for stage, record in state.get("stages", {}).items():
            if stage in WORKFLOW_ALIAS_STAGES:
                continue
            counts[str(record.get("status", "pending"))] += 1
    output_html.write_text(_dashboard_html(rows, counts, stages), encoding="utf-8")
    return output_html


def print_state_report(root: Path) -> dict[str, Any]:
    sessions = discover_sessions(root)
    stage_counts: dict[str, Counter] = {}
    for session in sessions:
        for stage, record in read_pipeline_state(session).get("stages", {}).items():
            if stage in WORKFLOW_ALIAS_STAGES:
                continue
            stage_counts.setdefault(stage, Counter())[str(record.get("status", "pending"))] += 1
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


def _stage_command(*, session: Path, dataset: str, stage: str, config: Path | None) -> list[str]:
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


def _session_row(session: Path, stages_to_show: list[str]) -> str:
    state = read_pipeline_state(session)
    stages = state.get("stages", {})
    cells = []
    for stage in stages_to_show:
        record = stages.get(stage) or {"status": "pending"}
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


def _stage_title(stage: str, record: dict[str, Any]) -> str:
    parts = [stage]
    if record.get("duration_s") is not None:
        parts.append(f"duration_s={record['duration_s']}")
    if record.get("derived_from"):
        parts.append(f"derived_from={record['derived_from']}")
    if record.get("error"):
        parts.append(str(record["error"]))
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
