#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.filtering import EgoHandFilterConfig, recover_missing_ego_hands  # noqa: E402
from scripts.run_ropedia_hamer_ego_filter import draw_candidates, make_contact_sheet  # noqa: E402


def resolve_path(path: str, base_dir: Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = base_dir / out
    return out.resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover missing ego_left/ego_right from rejected HaMeR candidates without rerunning HaMeR."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--recovery_min_area_ratio", type=float, default=0.001)
    parser.add_argument("--recovery_min_center_y_ratio", type=float, default=0.35)
    parser.add_argument("--recovery_max_median_depth_m", type=float, default=1.2)
    parser.add_argument("--contact_sheet_columns", type=int, default=4)
    parser.add_argument("--contact_sheet_thumb_width", type=int, default=360)
    return parser.parse_args()


def recover_frame(frame: Dict[str, Any], config: EgoHandFilterConfig) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = [dict(candidate) for candidate in frame.get("candidates", [])]
    selected = {
        "ego_left": None,
        "ego_right": None,
    }
    for idx, candidate in enumerate(candidates):
        if candidate.get("status") in {"ego_left", "ego_left_recovered"}:
            selected["ego_left"] = idx
        elif candidate.get("status") in {"ego_right", "ego_right_recovered"}:
            selected["ego_right"] = idx

    before = dict(selected)
    recover_missing_ego_hands(candidates, selected, config)
    counts: Dict[str, int] = {}
    for candidate in candidates:
        status = str(candidate.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1

    out = dict(frame)
    out["candidates"] = candidates
    out["selected"] = selected
    out["counts"] = counts
    out["recovery_summary"] = {
        "before": before,
        "after": selected,
        "recovered_left": before["ego_left"] is None and selected["ego_left"] is not None,
        "recovered_right": before["ego_right"] is None and selected["ego_right"] is not None,
    }
    return out


def main() -> None:
    cwd = Path.cwd()
    args = parse_args()
    input_json = resolve_path(args.input_json, cwd)
    output_dir = resolve_path(args.output_dir, cwd)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated_recovered"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    config_data = dict(data.get("filter_config", {}))
    config_data.update(
        {
            "recovery_min_area_ratio": args.recovery_min_area_ratio,
            "recovery_min_center_y_ratio": args.recovery_min_center_y_ratio,
            "recovery_max_median_depth_m": args.recovery_max_median_depth_m,
        }
    )
    known_config_fields = EgoHandFilterConfig.__dataclass_fields__.keys()
    config = EgoHandFilterConfig(
        **{key: value for key, value in config_data.items() if key in known_config_fields}
    )

    recovered_frames = []
    annotated_paths: List[Path] = []
    for frame in data.get("frames", []):
        recovered = recover_frame(frame, config)
        image_path = Path(recovered.get("image", ""))
        if image_path.exists():
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is not None:
                annotated = draw_candidates(image_bgr, recovered["candidates"])
                annotated_path = annotated_dir / f"frame_{int(recovered['frame_idx']):06d}_ego_filter_recovered.jpg"
                cv2.imwrite(str(annotated_path), annotated)
                recovered["annotated_recovered_image"] = str(annotated_path)
                annotated_paths.append(annotated_path)
        recovered_frames.append(recovered)

    data["frames"] = recovered_frames
    data["filter_config"] = config.__dict__
    outputs = dict(data.get("outputs", {}))
    outputs["recovered_candidates_json"] = str(output_dir / "ego_hand_candidates_recovered.json")
    outputs["recovered_contact_sheet"] = str(output_dir / "contact_sheet_ego_filter_recovered.jpg")
    outputs["annotated_recovered_dir"] = str(annotated_dir)
    data["outputs"] = outputs

    output_json = output_dir / "ego_hand_candidates_recovered.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(jsonable(data), f, ensure_ascii=False, indent=2)

    contact_sheet = output_dir / "contact_sheet_ego_filter_recovered.jpg"
    make_contact_sheet(
        annotated_paths,
        contact_sheet,
        columns=args.contact_sheet_columns,
        thumb_width=args.contact_sheet_thumb_width,
    )

    recovered_left = sum(1 for frame in recovered_frames if frame["recovery_summary"]["recovered_left"])
    recovered_right = sum(1 for frame in recovered_frames if frame["recovery_summary"]["recovered_right"])
    remaining_missing_left = [
        frame["frame_idx"] for frame in recovered_frames if frame["selected"].get("ego_left") is None
    ]
    remaining_missing_right = [
        frame["frame_idx"] for frame in recovered_frames if frame["selected"].get("ego_right") is None
    ]
    print(f"Recovered left frames: {recovered_left}")
    print(f"Recovered right frames: {recovered_right}")
    print(f"Remaining missing left: {remaining_missing_left}")
    print(f"Remaining missing right: {remaining_missing_right}")
    print(f"JSON: {output_json}")
    print(f"Contact sheet: {contact_sheet}")

    hamer_overlay_dir = data.get("outputs", {}).get("hamer_overlay_dir")
    if hamer_overlay_dir:
        source = Path(hamer_overlay_dir)
        target = output_dir / "hamer_overlay"
        if source.exists() and not target.exists():
            shutil.copytree(source, target)


if __name__ == "__main__":
    main()
