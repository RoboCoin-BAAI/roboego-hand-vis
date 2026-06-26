from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from roboego_hand_vis.egohand.temporal_tracking import SIDES
from scripts.export_depth_pointcloud import read_rgb_frame


def status_color_rgba(side: str, status: str, has_anomaly: bool) -> List[int]:
    if has_anomaly:
        return [255, 60, 60, 205]
    if status == "interpolated":
        return [80, 150, 255, 180]
    if status == "recovered":
        return [255, 220, 60, 190]
    if side == "ego_left":
        return [40, 230, 90, 200]
    return [255, 150, 40, 200]


def draw_debug_frame(rgb: np.ndarray, frame_idx: int, observations: Dict[str, Optional[Dict[str, Any]]]) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    y = 22
    cv2.putText(out, f"frame {frame_idx}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 24
    for side in SIDES:
        obs = observations.get(side)
        if obs is None:
            label = f"{side}: missing"
            color = (0, 0, 255)
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in obs.get("bbox_xyxy", [0, 0, 0, 0])]
            has_anomaly = bool([f for f in obs.get("anomaly_flags", []) if f != "interpolated"])
            rgba = status_color_rgba(side, obs.get("status", ""), has_anomaly)
            color = (rgba[2], rgba[1], rgba[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = (
                f"{side}: {obs.get('status')} "
                f"jump={obs.get('smoothed_translation_jump_m')} "
                f"flags={','.join(obs.get('anomaly_flags', [])) or 'ok'}"
            )
        cv2.putText(out, label[:95], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 20
    return out


def make_contact_sheet(image_paths: List[Path], output_path: Path, columns: int, thumb_width: int) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = thumb_width / float(image.shape[1])
        thumbs.append(cv2.resize(image, (thumb_width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    rows = int(np.ceil(len(thumbs) / columns))
    thumb_h = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows * thumb_h, columns * thumb_width, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r = i // columns
        c = i % columns
        y = r * thumb_h
        x = c * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    cv2.imwrite(str(output_path), sheet)


def export_debug_contact_sheet(
    tracks: Dict[str, List[Optional[Dict[str, Any]]]],
    video_path: Path,
    output_dir: Path,
    start: int,
    max_debug_frames: int,
) -> None:
    debug_dir = output_dir / "debug_frames"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    selected_offsets = []
    for offset in range(len(next(iter(tracks.values())))):
        observations = {side: tracks[side][offset] for side in SIDES}
        important = any(obs is None for obs in observations.values())
        important = important or any(
            obs is not None and (obs.get("status") == "interpolated" or obs.get("anomaly_flags"))
            for obs in observations.values()
        )
        if important or offset % 30 == 0:
            selected_offsets.append(offset)
    selected_offsets = selected_offsets[:max_debug_frames]

    paths = []
    for offset in selected_offsets:
        frame_idx = start + offset
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        panel = draw_debug_frame(rgb, frame_idx, {side: tracks[side][offset] for side in SIDES})
        out_path = debug_dir / f"frame_{frame_idx:06d}_tracking_debug.jpg"
        cv2.imwrite(str(out_path), panel)
        paths.append(out_path)
    make_contact_sheet(paths, output_dir / "contact_sheet_tracking_debug.jpg", columns=4, thumb_width=360)
