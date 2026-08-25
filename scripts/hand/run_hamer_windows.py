#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HaMeR ego filtering for every semantic hand window.")
    parser.add_argument("--windows_json", required=True)
    parser.add_argument("--manifest", default=None, help="Single-segment manifest. Omit when using --segments_root.")
    parser.add_argument("--depth_dir", default=None, help="Single-segment FoundationStereo depth dir. Omit when using --segments_root.")
    parser.add_argument("--output_root", default=None, help="Single output root. Omit when using --segments_root.")
    parser.add_argument("--segments_root", default=None, help="Session segments directory. Dispatches windows by source_segment_id.")
    parser.add_argument("--output_subdir", default="hamer_windows", help="Subdirectory under each materialized hand window when using --segments_root.")
    parser.add_argument("--hamer_root", required=True)
    parser.add_argument("--stage1_actions", required=True)
    parser.add_argument("--body_detector", default="regnety")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run each HaMeR window.")
    parser.add_argument("--save_mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine_mode", default="strict_ego_audit")
    parser.add_argument("--wearer_roi_x_margin_ratio", type=float, default=None)
    parser.add_argument("--wearer_roi_y_margin_ratio", type=float, default=None)
    parser.add_argument("--only_window_ids", nargs="*", default=None)
    parser.add_argument("--only_segment_ids", nargs="*", default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--skip_missing_segments", action="store_true")
    parser.add_argument(
        "--accept_uncertain_refined",
        action="store_true",
        help="Allow overlay-confirmed unstable candidates to be selected by the tolerant refine stage.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_window_command_plan(
    *,
    windows_json: Path,
    manifest: str | None,
    depth_dir: str | None,
    output_root: Path | None,
    segments_root: Path | None = None,
    output_subdir: str = "hamer_windows",
    hamer_root: str,
    stage1_actions: str,
    body_detector: str,
    batch_size: int,
    save_mesh: bool,
    python_executable: str,
    refine_mode: str = "strict_ego_audit",
    only_window_ids: list[str] | None,
    only_segment_ids: list[str] | None = None,
    skip_missing_segments: bool = False,
    accept_uncertain_refined: bool = False,
    wearer_roi_x_margin_ratio: float | None = None,
    wearer_roi_y_margin_ratio: float | None = None,
    max_windows: int | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(windows_json).expanduser().read_text(encoding="utf-8"))
    selected_ids = set(only_window_ids or [])
    selected_segment_ids = set(only_segment_ids or [])
    plan: list[dict[str, Any]] = []
    for window in payload.get("windows", []) or []:
        if not isinstance(window, dict):
            continue
        window_id = str(window.get("window_id", f"handwin_{len(plan):04d}"))
        if selected_ids and window_id not in selected_ids:
            continue
        source_segment_id = str(window.get("source_segment_id") or "")
        if selected_segment_ids and source_segment_id not in selected_segment_ids:
            continue
        frame_range = window.get("segment_frame_range") or {}
        start = int(frame_range.get("start", 0))
        num_frames = int(frame_range.get("num_frames", 0))
        if num_frames <= 0:
            continue
        end = start + num_frames - 1
        window_type = str(window.get("window_type", "hand_window"))
        hands = "_".join(str(hand) for hand in window.get("hands_required", []) or ["hand"])
        policy = str(window.get("recommended_stage1_hand_policy") or ("permissive" if window_type == "two_hand" else "conservative"))
        stage1_time_offset_s = float(window.get("start", (window.get("source_segment_range") or {}).get("start", 0.0)))
        try:
            window_io = _resolve_window_io(
                window=window,
                manifest=manifest,
                depth_dir=depth_dir,
                output_root=output_root,
                segments_root=segments_root,
                output_subdir=output_subdir,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if skip_missing_segments:
                plan.append(
                    {
                        "window_id": window_id,
                        "source_segment_id": source_segment_id,
                        "skipped": True,
                        "skip_reason": str(exc),
                    }
                )
                continue
            raise
        if window_io.get("start") is not None:
            start = int(window_io["start"])
        if window_io.get("num_frames") is not None:
            num_frames = int(window_io["num_frames"])
        end = start + num_frames - 1
        output_dir = window_io["output_root"] / f"{window_id}_{window_type}_{hands}_{start:06d}_{end:06d}"
        plan.append(
            {
                "window_id": window_id,
                "window_type": window_type,
                "source_segment_id": window.get("source_segment_id"),
                "segment_dir": str(window_io["segment_dir"]) if window_io["segment_dir"] else None,
                "skipped": False,
                "hands_required": window.get("hands_required", []),
                "start": start,
                "num_frames": num_frames,
                "end": end,
                "stage1_hand_policy": policy,
                "stage1_time_offset_s": stage1_time_offset_s,
                "depth_dir": str(window_io["depth_dir"]) if window_io["depth_dir"] else None,
                "depth_missing_for_hamer_only": window_io["depth_dir"] is None,
                "output_dir": str(output_dir),
                "command": _window_command(
                    manifest=str(window_io["manifest"]),
                    depth_dir=str(window_io["depth_dir"]) if window_io["depth_dir"] else None,
                    output_dir=str(output_dir),
                    hamer_root=hamer_root,
                    stage1_actions=stage1_actions,
                    start=start,
                    num_frames=num_frames,
                    body_detector=body_detector,
                    batch_size=batch_size,
                    save_mesh=save_mesh,
                    refine_mode=refine_mode,
                    python_executable=python_executable,
                    stage1_hand_policy=policy,
                    stage1_time_offset_s=stage1_time_offset_s,
                    accept_uncertain_refined=accept_uncertain_refined,
                    wearer_roi_x_margin_ratio=wearer_roi_x_margin_ratio,
                    wearer_roi_y_margin_ratio=wearer_roi_y_margin_ratio,
                ),
            }
        )
        if max_windows is not None and len(plan) >= max_windows:
            break
    return plan


def _resolve_window_io(
    *,
    window: dict[str, Any],
    manifest: str | None,
    depth_dir: str | None,
    output_root: Path | None,
    segments_root: Path | None,
    output_subdir: str,
) -> dict[str, Path | None]:
    if segments_root is None:
        if manifest is None or depth_dir is None or output_root is None:
            raise ValueError("Either provide --segments_root or provide --manifest, --depth_dir, and --output_root.")
        return {
            "manifest": Path(manifest),
            "depth_dir": Path(depth_dir),
            "output_root": output_root,
            "segment_dir": None,
            "start": None,
            "num_frames": None,
        }
    segment_id = str(window.get("source_segment_id") or "").strip()
    if not segment_id:
        raise ValueError(f"Window missing source_segment_id: {window.get('window_id')}")
    segment_dir = _find_segment_dir(Path(segments_root), segment_id)
    hand_window_dir = _find_hand_window_dir(segment_dir, str(window.get("window_id") or ""))
    if hand_window_dir is None:
        raise FileNotFoundError(
            f"Missing materialized hand-window directory for {window.get('window_id')} under {segment_dir / 'hand_windows'}"
        )
    manifest_path = hand_window_dir / "dataset_manifest/dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing hand-window manifest for {window.get('window_id')}: {manifest_path}")
    depth_path = _preferred_depth_dir(hand_window_dir)
    return {
        "manifest": manifest_path,
        "depth_dir": depth_path,
        "output_root": hand_window_dir / output_subdir,
        "segment_dir": segment_dir,
        "start": 0,
        "num_frames": int((window.get("segment_frame_range") or {}).get("num_frames", 0)),
    }


def _find_segment_dir(segments_root: Path, segment_id: str) -> Path:
    matches = sorted(path for path in segments_root.glob(f"{segment_id}_*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"No segment directory for {segment_id} under {segments_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple segment directories for {segment_id} under {segments_root}: {matches}")
    return matches[0]


def _find_hand_window_dir(segment_dir: Path, window_id: str) -> Path | None:
    if not window_id:
        return None
    hand_windows_root = segment_dir / "hand_windows"
    if not hand_windows_root.exists():
        return None
    exact = hand_windows_root / window_id
    if exact.is_dir():
        return exact
    matches = sorted(path for path in hand_windows_root.glob(f"{window_id}_*") if path.is_dir())
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Multiple hand-window directories for {window_id} under {hand_windows_root}: {matches}")
    return matches[0]


def _preferred_depth_dir(hand_window_dir: Path) -> Path | None:
    for name in ("fast_foundation_depth", "foundation_depth"):
        path = hand_window_dir / name
        if path.exists():
            return path
    return None


def _window_command(
    *,
    manifest: str,
    depth_dir: str,
    output_dir: str,
    hamer_root: str,
    stage1_actions: str,
    start: int,
    num_frames: int,
    body_detector: str,
    batch_size: int,
    save_mesh: bool,
    refine_mode: str,
    python_executable: str,
    stage1_hand_policy: str,
    stage1_time_offset_s: float = 0.0,
    accept_uncertain_refined: bool = False,
    wearer_roi_x_margin_ratio: float | None = None,
    wearer_roi_y_margin_ratio: float | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(PROJECT_ROOT / "scripts/hand/run_hamer_ego_filter.py"),
        "--manifest",
        manifest,
        "--output_dir",
        output_dir,
        "--hamer_root",
        hamer_root,
        "--start",
        str(start),
        "--num_frames",
        str(num_frames),
        "--body_detector",
        body_detector,
        "--batch_size",
        str(batch_size),
        "--ego_mode",
        "tolerant",
        "--stage1_actions",
        stage1_actions,
        "--stage1_hand_policy",
        stage1_hand_policy,
        "--stage1_time_offset_s",
        str(stage1_time_offset_s),
        "--wearer_regions_json",
        stage1_actions,
        "--refine_mode",
        refine_mode,
    ]
    if depth_dir:
        command.extend(["--depth_dir", depth_dir])
    if wearer_roi_x_margin_ratio is not None:
        command.extend(["--wearer_roi_x_margin_ratio", str(wearer_roi_x_margin_ratio)])
    if wearer_roi_y_margin_ratio is not None:
        command.extend(["--wearer_roi_y_margin_ratio", str(wearer_roi_y_margin_ratio)])
    if save_mesh:
        command.append("--save_mesh")
    if accept_uncertain_refined:
        command.append("--accept_uncertain_refined")
    return command


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser() if args.output_root else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    segments_root = Path(args.segments_root).expanduser() if args.segments_root else None
    plan = build_window_command_plan(
        windows_json=Path(args.windows_json).expanduser(),
        manifest=args.manifest,
        depth_dir=args.depth_dir,
        output_root=output_root,
        segments_root=segments_root,
        output_subdir=args.output_subdir,
        hamer_root=args.hamer_root,
        stage1_actions=args.stage1_actions,
        body_detector=args.body_detector,
        batch_size=args.batch_size,
        save_mesh=args.save_mesh,
        refine_mode=args.refine_mode,
        python_executable=args.python,
        only_window_ids=args.only_window_ids,
        only_segment_ids=args.only_segment_ids,
        skip_missing_segments=args.skip_missing_segments,
        accept_uncertain_refined=args.accept_uncertain_refined,
        wearer_roi_x_margin_ratio=args.wearer_roi_x_margin_ratio,
        wearer_roi_y_margin_ratio=args.wearer_roi_y_margin_ratio,
        max_windows=args.max_windows,
    )
    plan_root = output_root or segments_root
    if plan_root is None:
        raise ValueError("Either --output_root or --segments_root must be provided.")
    plan_path = plan_root / "hamer_window_run_plan.json"
    run_count = sum(1 for item in plan if not item.get("skipped"))
    skipped_count = sum(1 for item in plan if item.get("skipped"))
    plan_path.write_text(
        json.dumps(
            {
                "windows_json": args.windows_json,
                "window_count": len(plan),
                "run_count": run_count,
                "skipped_count": skipped_count,
                "runs": plan,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"plan": str(plan_path), "window_count": len(plan), "run_count": run_count, "skipped_count": skipped_count, "dry_run": args.dry_run}, indent=2, ensure_ascii=False))
    if args.dry_run:
        for item in plan:
            if item.get("skipped"):
                print(f"SKIP {item.get('window_id')} {item.get('source_segment_id')}: {item.get('skip_reason')}")
                continue
            print(" ".join(item["command"]))
        return
    for item in plan:
        if item.get("skipped"):
            print(f"skip {item.get('window_id')} {item.get('source_segment_id')}: {item.get('skip_reason')}")
            continue
        output_dir = Path(item["output_dir"])
        done_json = output_dir / "ego_hand_candidates_refined.json"
        if done_json.exists() and not args.overwrite:
            print(f"skip existing {item['window_id']}: {done_json}")
            continue
        print(f"run {item['window_id']} {item['window_type']} frames {item['start']}-{item['end']} policy={item['stage1_hand_policy']}")
        subprocess.run(item["command"], cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
