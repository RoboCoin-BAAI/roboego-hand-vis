#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.pipeline_state import discover_sessions, read_pipeline_state


@dataclass(frozen=True)
class MissingDepthWindow:
    session: Path
    unit: Path
    stereo_report: Path


def build_depth_command(
    *,
    python_bin: str,
    dataset: str,
    session: Path,
    hand_window: Path,
    config: Path | None,
) -> list[str]:
    cmd = [
        python_bin,
        "scripts/run_pipeline_stage.py",
        "--dataset",
        dataset,
        "--session",
        str(Path(session).expanduser().resolve()),
        "--stage",
        "fast_foundation_depth",
        "--hand-window",
        str(Path(hand_window).expanduser().resolve()),
        "--force",
    ]
    if config is not None:
        cmd.extend(["--config", str(config)])
    return cmd


def find_missing_depth_windows(root: Path) -> list[MissingDepthWindow]:
    windows: list[MissingDepthWindow] = []
    for session in discover_sessions(Path(root).expanduser().resolve()):
        state = read_pipeline_state(session)
        stages = state.get("stages", {}) if isinstance(state.get("stages"), dict) else {}
        if _stage_status(stages, "prepare_stereo") != "success":
            continue
        if _stage_status(stages, "stereo_summary_quality") != "success":
            continue
        if _stage_status(stages, "fast_foundation_depth") != "blocked":
            continue
        for unit in _prepared_units(session):
            if _depth_complete(unit):
                continue
            report = unit / "quality/stereo_summary/stereo_quality_report.json"
            if report.is_file():
                windows.append(MissingDepthWindow(session=session.resolve(), unit=unit.resolve(), stereo_report=report))
    return windows


def patch_depth_usable_report(unit: Path) -> bool:
    report = Path(unit) / "quality/stereo_summary/stereo_quality_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return False
    changed = False
    if summary.get("usable_for_depth") is not True:
        summary["usable_for_depth"] = True
        changed = True
    if summary.get("depth_risk_level") != "high_risk":
        summary["depth_risk_level"] = "high_risk"
        changed = True

    next_gate = payload.get("next_step_gate_zh")
    if isinstance(next_gate, dict):
        if next_gate.get("usable_for_depth") is not True:
            next_gate["usable_for_depth"] = True
            changed = True
        if next_gate.get("depth_risk_level") != "high_risk":
            next_gate["depth_risk_level"] = "high_risk"
            changed = True

    if changed:
        report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def run_missing_depth_windows(
    *,
    root: Path,
    dataset: str,
    python_bin: str,
    config: Path | None,
    dry_run: bool,
    patch_reports: bool,
    limit: int,
    exit_on_error: bool,
) -> dict[str, Any]:
    windows = find_missing_depth_windows(root)
    if limit > 0:
        windows = windows[:limit]
    results: list[dict[str, Any]] = []
    patched = 0
    for item in windows:
        if patch_reports and patch_depth_usable_report(item.unit):
            patched += 1
        cmd = build_depth_command(
            python_bin=python_bin,
            dataset=dataset,
            session=item.session,
            hand_window=item.unit,
            config=config,
        )
        if dry_run:
            print(" ".join(cmd))
            results.append({"session": str(item.session), "unit": str(item.unit), "status": "dry_run"})
            continue
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        status = "success" if completed.returncode == 0 else "blocked" if completed.returncode == 2 else "fail"
        results.append(
            {
                "session": str(item.session),
                "unit": str(item.unit),
                "status": status,
                "returncode": int(completed.returncode),
            }
        )
        if completed.returncode != 0 and exit_on_error:
            break
    return {
        "root": str(Path(root).expanduser().resolve()),
        "dataset": dataset,
        "candidate_count": len(windows),
        "patched_reports": patched,
        "status_counts": dict(Counter(result["status"] for result in results)),
        "results": results,
    }


def _stage_status(stages: dict[str, Any], stage: str) -> str:
    record = stages.get(stage)
    if not isinstance(record, dict):
        return "missing"
    return str(record.get("status") or "missing")


def _prepared_units(session: Path) -> list[Path]:
    units: list[Path] = []
    session = Path(session)
    if (session / "dataset_manifest/dataset_manifest.json").is_file():
        units.append(session)
    direct_root = session / "hand_windows"
    if direct_root.is_dir():
        units.extend(path.parents[1] for path in direct_root.glob("handwin_*/dataset_manifest/dataset_manifest.json"))
    segments_root = session / "segments"
    if segments_root.is_dir():
        units.extend(path.parents[1] for path in segments_root.glob("*/hand_windows/handwin_*/dataset_manifest/dataset_manifest.json"))
    return sorted({unit.resolve() for unit in units})


def _depth_complete(unit: Path) -> bool:
    depth_dir = Path(unit) / "depth/fast_foundation"
    meta = depth_dir / "fast_foundation_stereo_video_meta.json"
    png_dir = depth_dir / "depth_uint16_png"
    return meta.is_file() and png_dir.is_dir() and any(png_dir.glob("*_depth_mm.png"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch and rerun missing Fast-FoundationStereo depth hand windows.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="raw")
    parser.add_argument("--config", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-patch", action="store_true", help="Do not edit stereo summary JSON before running depth.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--exit-on-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_missing_depth_windows(
        root=Path(args.root),
        dataset=args.dataset,
        python_bin=args.python,
        config=Path(args.config).expanduser() if args.config else None,
        dry_run=bool(args.dry_run),
        patch_reports=not bool(args.no_patch),
        limit=max(0, int(args.limit)),
        exit_on_error=bool(args.exit_on_error),
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 1 if any(result["status"] == "fail" for result in summary["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
