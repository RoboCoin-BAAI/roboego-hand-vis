#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.missing_diagnostics import (  # noqa: E402
    candidate_label,
    missing_sides_from_frame,
    summarize_missing_frames,
)
from scripts.rerun_utils import depth_to_vis  # noqa: E402


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = base_dir / out
    return out.resolve()


def status_color(status: str) -> tuple[int, int, int]:
    if status == "ego_left":
        return (40, 220, 40)
    if status == "ego_right":
        return (40, 220, 40)
    if status in {"ego_left_recovered", "ego_right_recovered"}:
        return (0, 220, 255)
    if status == "other_hands":
        return (170, 170, 170)
    return (0, 0, 255)


def read_image(path: Path, fallback_shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is not None:
        return image
    h, w = fallback_shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(out, f"missing: {path.name}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    scale = width / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_header(image: np.ndarray, title: str, lines: Iterable[str] = ()) -> np.ndarray:
    line_list = [title, *[line for line in lines if line]]
    header_h = 28 + 20 * max(0, len(line_list) - 1)
    out = np.full((image.shape[0] + header_h, image.shape[1], 3), 245, dtype=np.uint8)
    out[header_h:, :] = image
    y = 20
    for i, line in enumerate(line_list):
        color = (0, 0, 0) if i == 0 else (70, 70, 70)
        scale = 0.55 if i == 0 else 0.48
        thickness = 2 if i == 0 else 1
        cv2.putText(out, line[:120], (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 20
    return out


def draw_candidate_details(image_bgr: np.ndarray, candidates: List[Dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    detail_lines = []
    for idx, candidate in enumerate(candidates):
        bbox = candidate.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        color = status_color(str(candidate.get("status", "")))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        short_label = (
            f"{candidate.get('candidate_id', idx)} "
            f"{candidate.get('status', '?')} "
            f"{candidate.get('handedness', '?')}"
        )
        text_org = (max(0, x1), max(18, y1 - 6))
        (tw, th), baseline = cv2.getTextSize(short_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            out,
            (text_org[0], text_org[1] - th - baseline - 2),
            (min(out.shape[1] - 1, text_org[0] + tw + 4), text_org[1] + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            out,
            short_label,
            (text_org[0] + 2, text_org[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        detail_lines.append(candidate_label(candidate))
        reasons = candidate.get("reject_reasons") or []
        if reasons:
            detail_lines.append(f"  reason={','.join(str(reason) for reason in reasons)}")

    if detail_lines:
        panel_h = 17 * len(detail_lines) + 12
        panel = np.full((panel_h, out.shape[1], 3), 255, dtype=np.uint8)
        for i, line in enumerate(detail_lines):
            cv2.putText(
                panel,
                line[:115],
                (8, 17 + i * 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        out = np.vstack([out, panel])
    return out


def depth_path_for(depth_dir: Path, frame_idx: int) -> Path:
    return depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"


def load_depth_vis(depth_dir: Path, frame_idx: int, min_m: float, max_m: float, fallback_shape: tuple[int, int]) -> np.ndarray:
    path = depth_path_for(depth_dir, frame_idx)
    if not path.exists():
        h, w = fallback_shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(out, f"missing depth: {path.name}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out
    depth = np.load(path).astype(np.float32)
    return cv2.cvtColor(depth_to_vis(depth, min_m, max_m), cv2.COLOR_RGB2BGR)


def make_row(images: List[np.ndarray], panel_width: int) -> np.ndarray:
    resized = [resize_to_width(image, panel_width) for image in images]
    max_h = max(image.shape[0] for image in resized)
    padded = []
    for image in resized:
        if image.shape[0] < max_h:
            pad = np.full((max_h - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
            image = np.vstack([image, pad])
        padded.append(image)
    return np.hstack(padded)


def make_contact_sheet(image_paths: List[Path], output_path: Path, columns: int, thumb_width: int) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        thumbs.append(resize_to_width(image, thumb_width))
    if not thumbs:
        return
    rows = int(np.ceil(len(thumbs) / columns))
    thumb_h = max(thumb.shape[0] for thumb in thumbs)
    sheet = np.full((rows * thumb_h, columns * thumb_width, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        row = i // columns
        col = i % columns
        y = row * thumb_h
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    cv2.imwrite(str(output_path), sheet)


def export_frame_panel(
    frame: Dict[str, Any],
    candidates_base_dir: Path,
    depth_dir: Path,
    output_dir: Path,
    panel_width: int,
    depth_min_m: float,
    depth_max_m: float,
) -> Path:
    frame_idx = int(frame["frame_idx"])
    missing = missing_sides_from_frame(frame)
    rgb_path = resolve_path(frame["image"], candidates_base_dir)
    overlay_path = resolve_path(frame.get("hamer_overlay_image", frame["image"]), candidates_base_dir)

    rgb = read_image(rgb_path)
    overlay = read_image(overlay_path, fallback_shape=rgb.shape[:2])
    candidates = draw_candidate_details(rgb, frame.get("candidates", []))
    depth_vis = load_depth_vis(depth_dir, frame_idx, depth_min_m, depth_max_m, rgb.shape[:2])

    selected = frame.get("selected") or {}
    summary_line = f"missing={','.join(missing)} selected_left={selected.get('ego_left')} selected_right={selected.get('ego_right')}"
    count_line = "counts=" + json.dumps(frame.get("counts", {}), ensure_ascii=False, sort_keys=True)

    row = make_row(
        [
            add_header(rgb, "RGB", [summary_line]),
            add_header(overlay, "HaMeR overlay"),
            add_header(candidates, "all candidates bbox + rejected reason", [count_line]),
            add_header(depth_vis, "depth_vis"),
        ],
        panel_width,
    )
    suffix = "_".join(side.replace("ego_", "") for side in missing)
    out_path = output_dir / "frames" / f"frame_{frame_idx:06d}_missing_{suffix}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), row)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export contact sheets for frames missing ego left/right hands.")
    parser.add_argument("--candidates_json", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--panel_width", type=int, default=360)
    parser.add_argument("--contact_sheet_columns", type=int, default=1)
    parser.add_argument("--contact_sheet_thumb_width", type=int, default=1600)
    parser.add_argument("--depth_min_m", type=float, default=0.15)
    parser.add_argument("--depth_max_m", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates_json = Path(args.candidates_json).expanduser().resolve()
    depth_dir = Path(args.depth_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(candidates_json.read_text())
    frames = data["frames"] if isinstance(data, dict) and "frames" in data else data
    summary = summarize_missing_frames(frames)
    candidates_base_dir = candidates_json.parent

    missing_frames = [frame for frame in frames if missing_sides_from_frame(frame)]
    frame_panels = [
        export_frame_panel(
            frame=frame,
            candidates_base_dir=candidates_base_dir,
            depth_dir=depth_dir,
            output_dir=output_dir,
            panel_width=args.panel_width,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
        )
        for frame in missing_frames
    ]

    left_panels = [
        path
        for frame, path in zip(missing_frames, frame_panels)
        if "ego_left" in missing_sides_from_frame(frame)
    ]
    right_panels = [
        path
        for frame, path in zip(missing_frames, frame_panels)
        if "ego_right" in missing_sides_from_frame(frame)
    ]

    make_contact_sheet(
        frame_panels,
        output_dir / "contact_sheet_missing_all.jpg",
        columns=args.contact_sheet_columns,
        thumb_width=args.contact_sheet_thumb_width,
    )
    make_contact_sheet(
        left_panels,
        output_dir / "contact_sheet_missing_left.jpg",
        columns=args.contact_sheet_columns,
        thumb_width=args.contact_sheet_thumb_width,
    )
    make_contact_sheet(
        right_panels,
        output_dir / "contact_sheet_missing_right.jpg",
        columns=args.contact_sheet_columns,
        thumb_width=args.contact_sheet_thumb_width,
    )

    summary.update(
        {
            "candidates_json": str(candidates_json),
            "depth_dir": str(depth_dir),
            "output_dir": str(output_dir),
            "frame_panels": [str(path) for path in frame_panels],
            "contact_sheet_missing_all": str(output_dir / "contact_sheet_missing_all.jpg"),
            "contact_sheet_missing_left": str(output_dir / "contact_sheet_missing_left.jpg"),
            "contact_sheet_missing_right": str(output_dir / "contact_sheet_missing_right.jpg"),
        }
    )
    (output_dir / "missing_hand_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"Missing any: {summary['missing_any_count']}")
    print(f"Missing left: {summary['missing_left_count']}")
    print(f"Missing right: {summary['missing_right_count']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
