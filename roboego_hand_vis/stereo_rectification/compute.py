#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from roboego_hand_vis.common.video_paths import find_named_video

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Undistort and stereo-rectify one session, then write dataset_manifest.json."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="阶段参数 YAML；默认使用当前 stage 目录下 recommended.yaml。",
    )
    parser.add_argument("--session", required=True, help="Session directory containing videos/left.mp4 and videos/right.mp4.")
    parser.add_argument("--roboegopipe_root", default="", help="旧参数兼容占位；当前实现不再依赖外部 RoboEgoPipe 仓库。")
    parser.add_argument("--dataset_adapter", default=None, help="Manifest adapter for rectified Wumei-style segments.")
    parser.add_argument("--model", choices=["double_sphere", "equidistant", "fisheye"], default=None)
    parser.add_argument("--balance", type=float, default=None, help="Equidistant/fisheye crop balance.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def resolved_options_from_yaml_and_args(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    io_config = data.get("io", {})
    parameters = data.get("parameters", {})

    return {
        "dataset_adapter": args.dataset_adapter or str(io_config.get("dataset_adapter", "wumei01")),
        "model": args.model or str(parameters.get("model", "equidistant")),
        "balance": float(args.balance if args.balance is not None else parameters.get("balance", 0.0)),
        "width": int(args.width if args.width is not None else parameters.get("width", 1600)),
        "height": int(args.height if args.height is not None else parameters.get("height", 900)),
        "overwrite": bool(args.overwrite if args.overwrite is not None else parameters.get("overwrite", False)),
        "dry_run": bool(args.dry_run if args.dry_run is not None else parameters.get("dry_run", False)),
    }


def main() -> None:
    args = parse_args()
    yaml_config = load_yaml(Path(args.config).expanduser().resolve())
    options = resolved_options_from_yaml_and_args(yaml_config, args)
    session = Path(args.session).expanduser().resolve()
    roboegopipe_root = Path(args.roboegopipe_root).expanduser().resolve() if args.roboegopipe_root else Path()
    item = prepare_stereo_unit(
        unit_root=session,
        session_dir=session,
        roboegopipe_root=roboegopipe_root,
        dataset_adapter=options["dataset_adapter"],
        model=options["model"],
        balance=options["balance"],
        width=options["width"],
        height=options["height"],
        overwrite=options["overwrite"],
        dry_run=options["dry_run"],
    )
    print(json.dumps({"session": str(session), "status": item.get("status"), "result": item}, indent=2, ensure_ascii=False))


def prepare_segment(
    *,
    segment_dir: Path,
    session_dir: Path,
    roboegopipe_root: Path,
    dataset_adapter: str,
    model: str,
    balance: float,
    width: int,
    height: int,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, str]:
    rectified = segment_dir / "rectified"
    work_dir = rectified / "_work"
    rectify_input = work_dir / "rectify_input"
    rectify_output_root = work_dir / "rectify_output"
    rectified_session = rectify_output_root / rectify_input.name
    manifest_dir = rectified / "dataset_manifest"
    out = {
        "segment_dir": str(segment_dir),
        "rectified_dir": str(rectified),
        "rectified_left": str(rectified / "videos/left_rectified.mp4"),
        "rectified_right": str(rectified / "videos/right_rectified.mp4"),
        "manifest": str(manifest_dir / "dataset_manifest.json"),
    }
    if dry_run:
        out["status"] = "dry_run"
        return out
    _validate_segment(segment_dir)
    if overwrite:
        _remove_known_outputs(rectified)
    _write_rectify_input(segment_dir=segment_dir, session_dir=session_dir, rectify_input=rectify_input)
    if not (rectified / "videos/left_rectified.mp4").exists() or not (rectified / "videos/right_rectified.mp4").exists():
        _run_rectify(
            rectify_input=rectify_input,
            rectify_output_root=rectify_output_root,
            roboegopipe_root=roboegopipe_root,
            model=model,
            balance=balance,
            width=width,
            height=height,
        )
        _move_rectified_outputs(rectified_session=rectified_session, rectified=rectified)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    _write_manifest(processed=rectified, manifest_dir=manifest_dir, segment_name=segment_dir.name, dataset_adapter=dataset_adapter)
    out["status"] = "ok"
    return out


def prepare_stereo_unit(
    *,
    unit_root: Path,
    session_dir: Path,
    roboegopipe_root: Path,
    dataset_adapter: str,
    model: str,
    balance: float,
    width: int,
    height: int,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, str]:
    unit_root = Path(unit_root)
    work_dir = unit_root / "_prepare_stereo_work"
    rectify_input = work_dir / "rectify_input"
    rectify_output_root = work_dir / "rectify_output"
    rectified_session = rectify_output_root / rectify_input.name
    manifest_dir = unit_root / "dataset_manifest"
    out = {
        "unit_root": str(unit_root),
        "rectified_left": str(unit_root / "videos/left_rectified.mp4"),
        "rectified_right": str(unit_root / "videos/right_rectified.mp4"),
        "manifest": str(manifest_dir / "dataset_manifest.json"),
    }
    if dry_run:
        out["status"] = "dry_run"
        return out
    _validate_segment(unit_root)
    if overwrite:
        _remove_stereo_unit_outputs(unit_root)
    _write_rectify_input(segment_dir=unit_root, session_dir=session_dir, rectify_input=rectify_input)
    if not (unit_root / "videos/left_rectified.mp4").exists() or not (unit_root / "videos/right_rectified.mp4").exists():
        _run_rectify(
            rectify_input=rectify_input,
            rectify_output_root=rectify_output_root,
            roboegopipe_root=roboegopipe_root,
            model=model,
            balance=balance,
            width=width,
            height=height,
        )
        _move_stereo_unit_outputs(rectified_session=rectified_session, unit_root=unit_root)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    _write_manifest(processed=unit_root, manifest_dir=manifest_dir, segment_name=unit_root.name, dataset_adapter=dataset_adapter)
    out["status"] = "ok"
    return out


def _selected_segment_dirs(segments_root: Path, segment_ids: list[str] | None) -> list[Path]:
    if not segments_root.exists():
        raise FileNotFoundError(f"Missing segments root: {segments_root}")
    dirs = sorted(path for path in segments_root.iterdir() if path.is_dir() and path.name.startswith("seg_"))
    if segment_ids:
        selected = set(segment_ids)
        dirs = [path for path in dirs if path.name.split("_t", 1)[0] in selected]
    if not dirs:
        raise FileNotFoundError(f"No selected segment dirs found under {segments_root}")
    return dirs


def _validate_segment(segment_dir: Path) -> None:
    for stream in ("left", "right"):
        find_named_video(segment_dir / "videos", stream)


def _remove_known_outputs(processed: Path) -> None:
    for name in ["videos", "calibrations", "timestamps", "dataset_manifest", "_work"]:
        path = processed / name
        if path.exists():
            shutil.rmtree(path)
    meta = processed / "meta.json"
    if meta.exists():
        meta.unlink()


def _remove_stereo_unit_outputs(unit_root: Path) -> None:
    for rel in ["videos/left_rectified.mp4", "videos/right_rectified.mp4"]:
        path = unit_root / rel
        if path.exists():
            path.unlink()
    for name in ["dataset_manifest", "_prepare_stereo_work"]:
        path = unit_root / name
        if path.exists():
            shutil.rmtree(path)


def _write_rectify_input(*, segment_dir: Path, session_dir: Path, rectify_input: Path) -> None:
    (rectify_input / "videos").mkdir(parents=True, exist_ok=True)
    (rectify_input / "calibrations").mkdir(parents=True, exist_ok=True)
    shutil.copy2(find_named_video(segment_dir / "videos", "left"), rectify_input / "videos/left.mp4")
    shutil.copy2(find_named_video(segment_dir / "videos", "right"), rectify_input / "videos/right.mp4")
    calibration_dir = segment_dir / "calibrations"
    if not calibration_dir.exists():
        calibration_dir = session_dir / "calibrations"
    if not calibration_dir.exists():
        raise FileNotFoundError(f"Missing session calibrations: {calibration_dir}")
    for path in calibration_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, rectify_input / "calibrations" / path.name)
    meta_path = session_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"fps": 30}
    meta["source_session"] = str(session_dir)
    meta["source_segment"] = str(segment_dir)
    meta["fps"] = meta.get("fps", 30)
    (rectify_input / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_rectify(
    *,
    rectify_input: Path,
    rectify_output_root: Path,
    roboegopipe_root: Path,
    model: str,
    balance: float,
    width: int,
    height: int,
) -> None:
    del roboegopipe_root
    command = [
        sys.executable,
        str(STAGE_DIR / "rectify_video.py"),
        "--input",
        str(rectify_input),
        "--output",
        str(rectify_output_root),
        "--model",
        model,
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    if model in {"equidistant", "fisheye"}:
        command.extend(["--balance", str(balance)])
    subprocess.run(command, cwd=str(STAGE_DIR), check=True)


def _move_rectified_outputs(*, rectified_session: Path, rectified: Path) -> None:
    if not rectified_session.exists():
        raise FileNotFoundError(f"Missing rectified output: {rectified_session}")
    for name in ["videos", "calibrations", "timestamps"]:
        src = rectified_session / name
        dst = rectified / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
    for source_video_name in ["left.mp4", "right.mp4"]:
        source_video = rectified / "videos" / source_video_name
        if source_video.exists():
            source_video.unlink()
    meta_src = rectified_session / "meta.json"
    if meta_src.exists():
        shutil.copy2(meta_src, rectified / "meta.json")


def _move_stereo_unit_outputs(*, rectified_session: Path, unit_root: Path) -> None:
    if not rectified_session.exists():
        raise FileNotFoundError(f"Missing rectified output: {rectified_session}")
    videos_dir = unit_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for name in ["left_rectified.mp4", "right_rectified.mp4"]:
        src = rectified_session / "videos" / name
        if src.exists():
            dst = videos_dir / name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
    for name in ["calibrations", "timestamps"]:
        src = rectified_session / name
        if src.exists():
            dst = unit_root / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
    meta_src = rectified_session / "meta.json"
    if meta_src.exists():
        shutil.copy2(meta_src, unit_root / "meta.json")


def _write_manifest(*, processed: Path, manifest_dir: Path, segment_name: str, dataset_adapter: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/datasets/resolve_dataset.py"),
            "--dataset",
            dataset_adapter,
            "--root",
            str(processed),
            "--output_dir",
            str(manifest_dir),
            "--sequence_name",
            segment_name,
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


if __name__ == "__main__":
    main()
