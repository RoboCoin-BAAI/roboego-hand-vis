#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare selected stereo segments for the standard pipeline: "
            "rectify/undistort with RoboEgoPipe and write dataset_manifest.json."
        )
    )
    parser.add_argument("--session", required=True, help="Session directory, e.g. data/session_20260804_224442")
    parser.add_argument(
        "--segments_root",
        default="",
        help="Selected segments directory. Defaults to <session>/segments.",
    )
    parser.add_argument(
        "--unit_root",
        default="",
        help="Prepare one stereo unit directly, e.g. a session, selected segment, or hand window root.",
    )
    parser.add_argument("--roboegopipe_root", default="/home/liuyou/Documents/RoboEgoPipe")
    parser.add_argument("--segment_ids", nargs="*", default=None, help="Optional source segment ids, e.g. seg_0001 seg_0002")
    parser.add_argument("--dataset_adapter", default="wumei01", help="Manifest adapter for rectified Wumei-style segments.")
    parser.add_argument("--model", choices=["double_sphere", "equidistant", "fisheye"], default="equidistant")
    parser.add_argument("--balance", type=float, default=0.0, help="Equidistant/fisheye crop balance.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = Path(args.session).expanduser().resolve()
    segments_root = Path(args.segments_root).expanduser().resolve() if args.segments_root else session / "segments"
    roboegopipe_root = Path(args.roboegopipe_root).expanduser().resolve()
    if args.unit_root:
        item = prepare_stereo_unit(
            unit_root=Path(args.unit_root).expanduser().resolve(),
            session_dir=session,
            roboegopipe_root=roboegopipe_root,
            dataset_adapter=args.dataset_adapter,
            model=args.model,
            balance=args.balance,
            width=args.width,
            height=args.height,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps({"session": str(session), "unit_count": 1, "units": [item]}, indent=2, ensure_ascii=False))
        return
    segment_dirs = _selected_segment_dirs(segments_root, args.segment_ids)
    summary: list[dict[str, str]] = []
    for segment_dir in segment_dirs:
        item = prepare_segment(
            segment_dir=segment_dir,
            session_dir=session,
            roboegopipe_root=roboegopipe_root,
            dataset_adapter=args.dataset_adapter,
            model=args.model,
            balance=args.balance,
            width=args.width,
            height=args.height,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        summary.append(item)
    print(json.dumps({"session": str(session), "segment_count": len(summary), "segments": summary}, indent=2, ensure_ascii=False))


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
    for rel in ["videos/left.mp4", "videos/right.mp4"]:
        path = segment_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing segment video: {path}")


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
    shutil.copy2(segment_dir / "videos/left.mp4", rectify_input / "videos/left.mp4")
    shutil.copy2(segment_dir / "videos/right.mp4", rectify_input / "videos/right.mp4")
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
    env = os.environ.copy()
    env["PYTHONPATH"] = str(roboegopipe_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        sys.executable,
        str(roboegopipe_root / "scripts/rectify_stereo_video.py"),
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
    subprocess.run(command, cwd=str(roboegopipe_root), env=env, check=True)


def _move_rectified_outputs(*, rectified_session: Path, rectified: Path) -> None:
    if not rectified_session.exists():
        raise FileNotFoundError(f"Missing RoboEgoPipe rectified output: {rectified_session}")
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
        raise FileNotFoundError(f"Missing RoboEgoPipe rectified output: {rectified_session}")
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
