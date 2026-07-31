# Object Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an object annotation step that segments the BAAI yellow object, labels it with Qwen, computes hand-object contact from depth/hand geometry, and exports reviewable JSON/Rerun artifacts.

**Architecture:** Build a new `roboego_hand_vis.objects` package with stable object observation dataclasses, backend-independent annotation IO, deterministic `color_yellow` segmentation, depth back-projection, contact computation, Qwen semantic labeling, and Rerun export. The first SAM3 integration is a mask-loader backend that consumes existing SAM3-style mask folders and emits the same observations as the color backend.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, SciPy KDTree, Rerun SDK, stdlib `urllib` for Qwen OpenAI-compatible HTTP calls, pytest.

---

## File Structure

- Create `roboego_hand_vis/objects/__init__.py`: package exports for object pipeline types and helpers.
- Create `roboego_hand_vis/objects/segmentation.py`: shared dataclasses and bbox/mask utilities.
- Create `roboego_hand_vis/objects/color_yellow.py`: deterministic BAAI yellow-object segmentation backend.
- Create `roboego_hand_vis/objects/sam3_backend.py`: loader for existing SAM3-style per-frame mask directories.
- Create `roboego_hand_vis/objects/depth_geometry.py`: depth loading, mask-depth stats, and pixel-to-camera point projection.
- Create `roboego_hand_vis/objects/contact.py`: fingertip/hand-object distance and contact annotation.
- Create `roboego_hand_vis/objects/semantic_qwen.py`: key-crop selection, strict JSON prompt, HTTP client, parser/cache.
- Create `roboego_hand_vis/objects/annotation_io.py`: JSON/CSV writers for object tracks, semantic labels, and contact.
- Create `roboego_hand_vis/objects/rerun_export.py`: RGB bbox/contact overlays and optional 3D object depth points.
- Create `scripts/object/run_object_annotation.py`: one CLI that runs segmentation, Qwen semantics, contact, and Rerun.
- Modify `README.md`: add the object annotation step and example commands.
- Add tests under `tests/objects/`.

## Task 1: Shared Object Types And Annotation IO

**Files:**
- Create: `roboego_hand_vis/objects/__init__.py`
- Create: `roboego_hand_vis/objects/segmentation.py`
- Create: `roboego_hand_vis/objects/annotation_io.py`
- Test: `tests/objects/test_annotation_io.py`

- [ ] **Step 1: Write failing tests for object observation serialization**

Create `tests/objects/test_annotation_io.py`:

```python
from pathlib import Path

import numpy as np

from roboego_hand_vis.objects.annotation_io import write_object_tracks_json
from roboego_hand_vis.objects.segmentation import ObjectObservation, bbox_xywh_to_xyxy, crop_with_padding


def test_bbox_xywh_to_xyxy():
    assert bbox_xywh_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]


def test_crop_with_padding_clips_to_image_bounds():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = crop_with_padding(image, [25, 15, 20, 20], padding_px=4)
    assert crop.shape == (9, 9, 3)


def test_write_object_tracks_json_uses_relative_artifact_paths(tmp_path: Path):
    obs = ObjectObservation(
        frame_idx=0,
        object_id="yellow_object_0",
        bbox_xywh=[1, 2, 3, 4],
        mask_area_px=12,
        source="test",
        mask_path=tmp_path / "masks/frame_000000_yellow_object_0.png",
        crop_path=tmp_path / "crops/frame_000000_yellow_object_0.png",
        depth_median_m=0.35,
        depth_valid_ratio=0.9,
    )
    out = tmp_path / "object_tracks.json"
    payload = write_object_tracks_json(
        observations={"yellow_object_0": [obs]},
        output_path=out,
        output_dir=tmp_path,
        backend="color_yellow",
        video_path=Path("video.mp4"),
        depth_dir=Path("depth"),
    )
    frame = payload["objects"]["yellow_object_0"]["frames"][0]
    assert frame["mask_path"] == "masks/frame_000000_yellow_object_0.png"
    assert frame["crop_path"] == "crops/frame_000000_yellow_object_0.png"
    assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/objects/test_annotation_io.py -q
```

Expected: import failure for `roboego_hand_vis.objects`.

- [ ] **Step 3: Implement shared types and JSON writer**

Create `roboego_hand_vis/objects/segmentation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ObjectObservation:
    frame_idx: int
    object_id: str
    bbox_xywh: Optional[list[int]]
    mask_area_px: int
    source: str
    mask_path: Optional[Path] = None
    crop_path: Optional[Path] = None
    depth_median_m: Optional[float] = None
    depth_valid_ratio: Optional[float] = None


def bbox_xywh_to_xyxy(bbox_xywh: list[int]) -> list[int]:
    x, y, w, h = [int(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]


def crop_with_padding(image: np.ndarray, bbox_xywh: list[int], padding_px: int) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox_xywh]
    height, width = image.shape[:2]
    x1 = max(0, x - padding_px)
    y1 = max(0, y - padding_px)
    x2 = min(width, x + w + padding_px)
    y2 = min(height, y + h + padding_px)
    return image[y1:y2, x1:x2].copy()
```

Create `roboego_hand_vis/objects/annotation_io.py`:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .segmentation import ObjectObservation


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def observation_to_json(obs: ObjectObservation, output_dir: Path) -> Dict[str, Any]:
    return {
        "frame_idx": int(obs.frame_idx),
        "bbox_xywh": obs.bbox_xywh,
        "mask_path": _rel(obs.mask_path, output_dir),
        "crop_path": _rel(obs.crop_path, output_dir),
        "mask_area_px": int(obs.mask_area_px),
        "depth_median_m": obs.depth_median_m,
        "depth_valid_ratio": obs.depth_valid_ratio,
        "source": obs.source,
    }


def write_object_tracks_json(
    *,
    observations: Dict[str, List[ObjectObservation]],
    output_path: Path,
    output_dir: Path,
    backend: str,
    video_path: Path,
    depth_dir: Path,
) -> Dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "backend": backend,
        "video": str(video_path),
        "depth_dir": str(depth_dir),
        "objects": {
            object_id: {"frames": [observation_to_json(obs, output_dir) for obs in frames]}
            for object_id, frames in observations.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def write_object_tracks_csv(observations: Dict[str, List[ObjectObservation]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "object_id",
                "bbox_xywh",
                "mask_area_px",
                "depth_median_m",
                "depth_valid_ratio",
                "source",
            ],
        )
        writer.writeheader()
        for object_id, frames in observations.items():
            for obs in frames:
                writer.writerow(
                    {
                        "frame_idx": obs.frame_idx,
                        "object_id": object_id,
                        "bbox_xywh": obs.bbox_xywh,
                        "mask_area_px": obs.mask_area_px,
                        "depth_median_m": obs.depth_median_m,
                        "depth_valid_ratio": obs.depth_valid_ratio,
                        "source": obs.source,
                    }
                )
```

Create `roboego_hand_vis/objects/__init__.py`:

```python
from .segmentation import ObjectObservation

__all__ = ["ObjectObservation"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/objects/test_annotation_io.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add roboego_hand_vis/objects/__init__.py roboego_hand_vis/objects/segmentation.py roboego_hand_vis/objects/annotation_io.py tests/objects/test_annotation_io.py
git commit -m "feat: add object annotation data types"
```

## Task 2: Deterministic Yellow Object Segmentation Backend

**Files:**
- Modify: `roboego_hand_vis/objects/segmentation.py`
- Create: `roboego_hand_vis/objects/color_yellow.py`
- Test: `tests/objects/test_color_yellow.py`

- [ ] **Step 1: Write failing tests for yellow segmentation and gap filling**

Create `tests/objects/test_color_yellow.py`:

```python
import numpy as np

from roboego_hand_vis.objects.color_yellow import (
    YellowSegmentationConfig,
    detect_yellow_object,
    fill_missing_observations,
)
from roboego_hand_vis.objects.segmentation import ObjectObservation


def test_detect_yellow_object_returns_largest_lower_roi_component():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[60:85, 40:80] = [255, 230, 20]
    image[5:20, 5:35] = [255, 230, 20]
    obs, mask = detect_yellow_object(
        image,
        frame_idx=3,
        object_id="yellow_object_0",
        config=YellowSegmentationConfig(roi_y_min_ratio=0.4, min_area_px=20),
    )
    assert obs.bbox_xywh == [40, 60, 40, 25]
    assert obs.mask_area_px == 1000
    assert mask.dtype == np.uint8


def test_fill_missing_observations_interpolates_short_gap():
    frames = [
        ObjectObservation(0, "yellow_object_0", [0, 0, 10, 10], 100, "detected"),
        ObjectObservation(1, "yellow_object_0", None, 0, "missing"),
        ObjectObservation(2, "yellow_object_0", [10, 10, 10, 10], 100, "detected"),
    ]
    filled = fill_missing_observations(frames, max_gap=3)
    assert filled[1].bbox_xywh == [5, 5, 10, 10]
    assert filled[1].source == "interpolated_yellow_bbox"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/objects/test_color_yellow.py -q
```

Expected: import failure for `roboego_hand_vis.objects.color_yellow`.

- [ ] **Step 3: Implement color segmentation backend**

Create `roboego_hand_vis/objects/color_yellow.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .segmentation import ObjectObservation, crop_with_padding


@dataclass
class YellowSegmentationConfig:
    hsv_lower: tuple[int, int, int] = (16, 55, 70)
    hsv_upper: tuple[int, int, int] = (45, 255, 255)
    roi_x_min_ratio: float = 0.20
    roi_x_max_ratio: float = 0.80
    roi_y_min_ratio: float = 0.38
    min_area_px: int = 900
    open_kernel_px: int = 5
    close_kernel_px: int = 17
    gap_fill_frames: int = 15


def detect_yellow_object(
    image_rgb: np.ndarray,
    *,
    frame_idx: int,
    object_id: str,
    config: YellowSegmentationConfig,
) -> tuple[ObjectObservation, np.ndarray]:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    height, width = image_rgb.shape[:2]
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[
        int(height * config.roi_y_min_ratio):height,
        int(width * config.roi_x_min_ratio):int(width * config.roi_x_max_ratio),
    ] = 255
    mask = cv2.inRange(hsv, np.array(config.hsv_lower), np.array(config.hsv_upper))
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((config.open_kernel_px, config.open_kernel_px), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((config.close_kernel_px, config.close_kernel_px), np.uint8))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[tuple[int, list[int], list[float], int]] = []
    for label in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[label]]
        cx, cy = [float(v) for v in centroids[label]]
        if area < config.min_area_px:
            continue
        components.append((area, [x, y, w, h], [cx, cy], label))
    if not components:
        return ObjectObservation(frame_idx, object_id, None, 0, "missing"), np.zeros_like(mask)
    components.sort(reverse=True)
    area, bbox, _centroid, label = components[0]
    object_mask = ((labels == label).astype(np.uint8) * 255)
    return ObjectObservation(frame_idx, object_id, bbox, int(area), "yellow_hsv_component"), object_mask


def fill_missing_observations(frames: list[ObjectObservation], max_gap: int) -> list[ObjectObservation]:
    valid = [idx for idx, obs in enumerate(frames) if obs.bbox_xywh is not None]
    out = list(frames)
    for idx, obs in enumerate(frames):
        if obs.bbox_xywh is not None:
            continue
        prev = max((i for i in valid if i < idx), default=None)
        nxt = min((i for i in valid if i > idx), default=None)
        if prev is None or nxt is None or nxt - prev > max_gap:
            continue
        ratio = (idx - prev) / (nxt - prev)
        a = np.asarray(frames[prev].bbox_xywh, dtype=np.float32)
        b = np.asarray(frames[nxt].bbox_xywh, dtype=np.float32)
        bbox = np.round(a * (1.0 - ratio) + b * ratio).astype(int).tolist()
        out[idx] = ObjectObservation(obs.frame_idx, obs.object_id, bbox, bbox[2] * bbox[3], "interpolated_yellow_bbox")
    return out


def save_mask_and_crop(
    *,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    obs: ObjectObservation,
    output_dir: Path,
    crop_padding_px: int,
) -> ObjectObservation:
    if obs.bbox_xywh is None:
        return obs
    masks_dir = output_dir / "masks"
    crops_dir = output_dir / "crops"
    masks_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{obs.frame_idx:06d}_{obs.object_id}"
    mask_path = masks_dir / f"{stem}.png"
    crop_path = crops_dir / f"{stem}.png"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(crop_path), cv2.cvtColor(crop_with_padding(image_rgb, obs.bbox_xywh, crop_padding_px), cv2.COLOR_RGB2BGR))
    obs.mask_path = mask_path
    obs.crop_path = crop_path
    return obs
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/objects/test_color_yellow.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add roboego_hand_vis/objects/color_yellow.py tests/objects/test_color_yellow.py
git commit -m "feat: add yellow object segmentation"
```

## Task 3: Depth Back-Projection And Object Depth Stats

**Files:**
- Create: `roboego_hand_vis/objects/depth_geometry.py`
- Test: `tests/objects/test_depth_geometry.py`

- [ ] **Step 1: Write failing tests for depth projection**

Create `tests/objects/test_depth_geometry.py`:

```python
import numpy as np

from roboego_hand_vis.objects.depth_geometry import MaskDepthStats, mask_depth_to_points


def test_mask_depth_to_points_projects_pixels_to_camera_frame():
    depth = np.ones((3, 3), dtype=np.float32) * 2.0
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255
    points, stats = mask_depth_to_points(
        depth,
        mask,
        intrinsics={"fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 1.0},
        min_depth_m=0.1,
        max_depth_m=5.0,
        stride=1,
    )
    assert isinstance(stats, MaskDepthStats)
    assert points.tolist() == [[0.0, 0.0, 2.0]]
    assert stats.depth_median_m == 2.0
    assert stats.depth_valid_ratio == 1.0


def test_mask_depth_to_points_filters_invalid_depth():
    depth = np.array([[np.nan, 1.0], [10.0, 2.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=np.uint8) * 255
    points, stats = mask_depth_to_points(
        depth,
        mask,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        min_depth_m=0.5,
        max_depth_m=3.0,
        stride=1,
    )
    assert len(points) == 2
    assert stats.valid_depth_count == 2
    assert stats.mask_pixel_count == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/objects/test_depth_geometry.py -q
```

Expected: import failure for `roboego_hand_vis.objects.depth_geometry`.

- [ ] **Step 3: Implement depth geometry helpers**

Create `roboego_hand_vis/objects/depth_geometry.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MaskDepthStats:
    mask_pixel_count: int
    valid_depth_count: int
    depth_valid_ratio: float
    depth_median_m: float | None
    depth_p10_m: float | None
    depth_p90_m: float | None


def depth_path_for_frame(depth_dir: Path, frame_idx: int) -> Path:
    return depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"


def load_depth_frame(depth_dir: Path, frame_idx: int) -> np.ndarray:
    path = depth_path_for_frame(depth_dir, frame_idx)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def mask_depth_to_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    *,
    intrinsics: dict[str, float],
    min_depth_m: float,
    max_depth_m: float,
    stride: int,
    max_points: int = 20000,
) -> tuple[np.ndarray, MaskDepthStats]:
    stride = max(1, int(stride))
    ys, xs = np.nonzero(mask[::stride, ::stride] > 0)
    ys = ys * stride
    xs = xs * stride
    mask_pixel_count = int((mask > 0).sum())
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32), MaskDepthStats(mask_pixel_count, 0, 0.0, None, None, None)
    z = depth_m[ys, xs].astype(np.float32)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32), MaskDepthStats(mask_pixel_count, 0, 0.0, None, None, None)
    if max_points > 0 and len(z) > max_points:
        keep = np.linspace(0, len(z) - 1, max_points).astype(np.int64)
        xs = xs[keep]
        ys = ys[keep]
        z = z[keep]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x3 = (xs.astype(np.float32) - cx) * z / fx
    y3 = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x3, y3, z], axis=1).astype(np.float32)
    all_valid_z = depth_m[mask > 0]
    all_valid_z = all_valid_z[np.isfinite(all_valid_z) & (all_valid_z >= min_depth_m) & (all_valid_z <= max_depth_m)]
    stats = MaskDepthStats(
        mask_pixel_count=mask_pixel_count,
        valid_depth_count=int(len(all_valid_z)),
        depth_valid_ratio=float(len(all_valid_z) / max(1, mask_pixel_count)),
        depth_median_m=float(np.median(all_valid_z)),
        depth_p10_m=float(np.percentile(all_valid_z, 10)),
        depth_p90_m=float(np.percentile(all_valid_z, 90)),
    )
    return points, stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/objects/test_depth_geometry.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add roboego_hand_vis/objects/depth_geometry.py tests/objects/test_depth_geometry.py
git commit -m "feat: add object depth geometry"
```

## Task 4: Hand-Object Contact Computation

**Files:**
- Create: `roboego_hand_vis/objects/contact.py`
- Test: `tests/objects/test_contact.py`

- [ ] **Step 1: Write failing tests for fingertip contact**

Create `tests/objects/test_contact.py`:

```python
import numpy as np

from roboego_hand_vis.objects.contact import (
    FINGERTIP_NAMES,
    compute_fingertip_contact,
    extract_fingertips_from_joints21,
)


def test_extract_fingertips_from_joints21_uses_openpose_indices():
    joints = np.zeros((21, 3), dtype=np.float32)
    for idx in [4, 8, 12, 16, 20]:
        joints[idx] = [idx, idx + 1, idx + 2]
    fingertips = extract_fingertips_from_joints21(joints)
    assert list(fingertips.keys()) == FINGERTIP_NAMES
    assert fingertips["index_tip"].tolist() == [8.0, 9.0, 10.0]


def test_compute_fingertip_contact_marks_close_tip():
    object_points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    fingertips = {
        "thumb_tip": np.array([0.01, 0.0, 0.0], dtype=np.float32),
        "index_tip": np.array([0.2, 0.0, 0.0], dtype=np.float32),
    }
    result = compute_fingertip_contact(fingertips, object_points, threshold_m=0.025)
    assert result.contact is True
    assert result.min_distance_m == 0.01
    assert result.contact_points == ["thumb_tip"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/objects/test_contact.py -q
```

Expected: import failure for `roboego_hand_vis.objects.contact`.

- [ ] **Step 3: Implement contact helpers**

Create `roboego_hand_vis/objects/contact.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGERTIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]


@dataclass
class ContactResult:
    contact: bool
    min_distance_m: float | None
    contact_points: list[str]


def extract_fingertips_from_joints21(joints21: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.asarray(joints21, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected joints21 shape (21, 3), got {arr.shape}")
    return {name: arr[idx] for name, idx in zip(FINGERTIP_NAMES, FINGERTIP_INDICES)}


def compute_fingertip_contact(
    fingertips: Dict[str, np.ndarray],
    object_points: np.ndarray,
    *,
    threshold_m: float,
) -> ContactResult:
    points = np.asarray(object_points, dtype=np.float32)
    if len(points) == 0 or not fingertips:
        return ContactResult(False, None, [])
    tree = cKDTree(points)
    contact_points: list[str] = []
    min_distance: float | None = None
    for name, tip in fingertips.items():
        dist, _ = tree.query(np.asarray(tip, dtype=np.float32), k=1)
        dist = float(dist)
        min_distance = dist if min_distance is None else min(min_distance, dist)
        if dist <= threshold_m:
            contact_points.append(name)
    return ContactResult(bool(contact_points), min_distance, contact_points)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/objects/test_contact.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add roboego_hand_vis/objects/contact.py tests/objects/test_contact.py
git commit -m "feat: add object contact geometry"
```

## Task 5: Qwen Semantic Annotation Parser And Client

**Files:**
- Create: `roboego_hand_vis/objects/semantic_qwen.py`
- Test: `tests/objects/test_semantic_qwen.py`

- [ ] **Step 1: Write failing tests for response parsing and key-frame selection**

Create `tests/objects/test_semantic_qwen.py`:

```python
from pathlib import Path

from roboego_hand_vis.objects.semantic_qwen import parse_semantic_response, select_key_frames
from roboego_hand_vis.objects.segmentation import ObjectObservation


def test_parse_semantic_response_accepts_json_object():
    parsed = parse_semantic_response(
        '{"category":"box","color":"yellow","shape":"rectangular","material":"paper","visual_description":"yellow box","confidence":0.8}'
    )
    assert parsed["category"] == "box"
    assert parsed["confidence"] == 0.8


def test_parse_semantic_response_extracts_json_from_text():
    parsed = parse_semantic_response(
        'result: {"category":"toy","color":"yellow","shape":"box","material":"plastic","visual_description":"toy","confidence":0.6}'
    )
    assert parsed["material"] == "plastic"


def test_select_key_frames_uses_first_largest_and_spaced_frames(tmp_path: Path):
    observations = [
        ObjectObservation(0, "yellow_object_0", [0, 0, 10, 10], 100, "detected", crop_path=tmp_path / "0.png"),
        ObjectObservation(10, "yellow_object_0", [0, 0, 30, 30], 900, "detected", crop_path=tmp_path / "10.png"),
        ObjectObservation(20, "yellow_object_0", [0, 0, 20, 20], 400, "detected", crop_path=tmp_path / "20.png"),
    ]
    assert select_key_frames(observations, max_frames=3) == [0, 10, 20]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/objects/test_semantic_qwen.py -q
```

Expected: import failure for `roboego_hand_vis.objects.semantic_qwen`.

- [ ] **Step 3: Implement parser, key-frame selection, and HTTP client**

Create `roboego_hand_vis/objects/semantic_qwen.py`:

```python
from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from .segmentation import ObjectObservation


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
REQUIRED_FIELDS = ["category", "color", "shape", "material", "visual_description", "confidence"]


def parse_semantic_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return {"status": "parse_failed", "raw_response": text}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"status": "parse_failed", "raw_response": text}
    out = {field: data.get(field) for field in REQUIRED_FIELDS}
    out["confidence"] = float(out["confidence"] or 0.0)
    out["status"] = "ok"
    return out


def select_key_frames(observations: list[ObjectObservation], max_frames: int) -> list[int]:
    valid = [obs for obs in observations if obs.bbox_xywh is not None and obs.crop_path is not None]
    if not valid:
        return []
    selected = {valid[0].frame_idx}
    largest = max(valid, key=lambda obs: obs.mask_area_px)
    selected.add(largest.frame_idx)
    if max_frames > len(selected):
        step = max(1, len(valid) // max(1, max_frames - len(selected)))
        for obs in valid[::step]:
            selected.add(obs.frame_idx)
            if len(selected) >= max_frames:
                break
    return sorted(selected)


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_qwen_messages(crop_paths: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Identify the manipulated object in these crops. Return strict JSON with keys: "
                "category, color, shape, material, visual_description, confidence. "
                "Use concise English values and confidence from 0 to 1."
            ),
        }
    ]
    for path in crop_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
    return [{"role": "user", "content": content}]


def call_qwen_semantics(
    *,
    crop_paths: list[Path],
    model: str,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    body = {
        "model": model,
        "messages": build_qwen_messages(crop_paths),
        "temperature": 0,
    }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = parse_semantic_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    return parsed
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/objects/test_semantic_qwen.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add roboego_hand_vis/objects/semantic_qwen.py tests/objects/test_semantic_qwen.py
git commit -m "feat: add qwen semantic annotation helpers"
```

## Task 6: SAM3-Style Mask Loader Backend

**Files:**
- Create: `roboego_hand_vis/objects/sam3_backend.py`
- Test: `tests/objects/test_sam3_backend.py`

- [ ] **Step 1: Write failing tests for SAM3 mask directory loading**

Create `tests/objects/test_sam3_backend.py`:

```python
from pathlib import Path

import cv2
import numpy as np

from roboego_hand_vis.objects.sam3_backend import load_sam3_mask_observation


def test_load_sam3_mask_observation_reads_frame_mask(tmp_path: Path):
    masks_root = tmp_path / "video_segmentation/masks"
    frame_dir = masks_root / "frame_000005_masks"
    frame_dir.mkdir(parents=True)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[6:10, 8:15] = 255
    cv2.imwrite(str(frame_dir / "cup.png"), mask)
    obs, loaded = load_sam3_mask_observation(masks_root, frame_idx=5, object_id="cup")
    assert obs.bbox_xywh == [8, 6, 7, 4]
    assert obs.mask_area_px == 28
    assert loaded.shape == (20, 30)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/objects/test_sam3_backend.py -q
```

Expected: import failure for `roboego_hand_vis.objects.sam3_backend`.

- [ ] **Step 3: Implement SAM3 mask loader**

Create `roboego_hand_vis/objects/sam3_backend.py`:

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .segmentation import ObjectObservation


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def sam3_mask_path(masks_root: Path, frame_idx: int, object_id: str) -> Path:
    return masks_root / f"frame_{frame_idx:06d}_masks" / f"{object_id}.png"


def load_sam3_mask_observation(
    masks_root: Path,
    *,
    frame_idx: int,
    object_id: str,
) -> tuple[ObjectObservation, np.ndarray]:
    path = sam3_mask_path(masks_root, frame_idx, object_id)
    if not path.exists():
        return ObjectObservation(frame_idx, object_id, None, 0, "sam3_mask_missing"), np.zeros((0, 0), dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    mask = ((mask > 0).astype(np.uint8) * 255)
    bbox = _bbox_from_mask(mask)
    area = int((mask > 0).sum())
    return ObjectObservation(frame_idx, object_id, bbox, area, "sam3_mask"), mask
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/objects/test_sam3_backend.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add roboego_hand_vis/objects/sam3_backend.py tests/objects/test_sam3_backend.py
git commit -m "feat: add sam3 mask loader backend"
```

## Task 7: Pipeline CLI, Contact JSON, And Rerun Export

**Files:**
- Create: `roboego_hand_vis/objects/rerun_export.py`
- Modify: `roboego_hand_vis/objects/annotation_io.py`
- Create: `scripts/object/run_object_annotation.py`
- Modify: `README.md`
- Test: `tests/objects/test_object_pipeline_cli.py`

- [ ] **Step 1: Write failing CLI smoke test**

Create `tests/objects/test_object_pipeline_cli.py`:

```python
from pathlib import Path

import cv2
import numpy as np

from scripts.object.run_object_annotation import run_annotation


def write_tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    for _ in range(3):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[28:40, 20:38] = [20, 230, 255]
        writer.write(frame)
    writer.release()


def test_run_annotation_offline_writes_core_outputs(tmp_path: Path):
    video = tmp_path / "video.mp4"
    write_tiny_video(video)
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    for idx in range(3):
        np.save(depth_dir / f"frame_{idx:06d}_foundation_depth_m.npy", np.ones((48, 64), dtype=np.float32) * 0.5)
    out = tmp_path / "out"
    run_annotation(
        video_path=video,
        depth_dir=depth_dir,
        output_dir=out,
        intrinsics={"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0},
        hand_tracks_path=None,
        hand_joints_npz=None,
        backend="color_yellow",
        object_id="yellow_object_0",
        num_frames=3,
        no_qwen=True,
        save_rrd=False,
    )
    assert (out / "object_tracks.json").exists()
    assert (out / "contact_annotations.json").exists()
    assert (out / "semantic_annotations.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/objects/test_object_pipeline_cli.py -q
```

Expected: import failure for `scripts.object.run_object_annotation`.

- [ ] **Step 3: Add contact JSON writers**

Append to `roboego_hand_vis/objects/annotation_io.py`:

```python
def write_json_payload(payload: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
```

- [ ] **Step 4: Implement Rerun overlay helpers**

Create `roboego_hand_vis/objects/rerun_export.py`:

```python
from __future__ import annotations

import cv2
import numpy as np
import rerun as rr

from .segmentation import ObjectObservation


def draw_object_bbox_overlay(
    image_rgb: np.ndarray,
    obs: ObjectObservation,
    *,
    label: str,
    contact_text: str | None = None,
) -> np.ndarray:
    overlay = image_rgb.copy()
    if obs.bbox_xywh is None:
        return overlay
    x, y, w, h = [int(v) for v in obs.bbox_xywh]
    color = (255, 40, 20)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 4)
    text = label if contact_text is None else f"{label} {contact_text}"
    cv2.putText(overlay, text, (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return overlay


def log_object_overlay(entity_path: str, image_rgb: np.ndarray) -> None:
    rr.log(entity_path, rr.Image(image_rgb))
```

- [ ] **Step 5: Implement CLI and callable `run_annotation`**

Create `scripts/object/run_object_annotation.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.objects.annotation_io import write_json_payload, write_object_tracks_csv, write_object_tracks_json
from roboego_hand_vis.objects.color_yellow import YellowSegmentationConfig, detect_yellow_object, fill_missing_observations, save_mask_and_crop
from roboego_hand_vis.objects.depth_geometry import load_depth_frame, mask_depth_to_points
from roboego_hand_vis.objects.rerun_export import draw_object_bbox_overlay, log_object_overlay
from roboego_hand_vis.objects.semantic_qwen import call_qwen_semantics, select_key_frames


def read_rgb_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_annotation(
    *,
    video_path: Path,
    depth_dir: Path,
    output_dir: Path,
    intrinsics: dict[str, float],
    hand_tracks_path: Path | None,
    hand_joints_npz: Path | None,
    backend: str,
    object_id: str,
    num_frames: int,
    no_qwen: bool,
    save_rrd: bool,
    qwen_model: str = "qwen-vl-plus",
    contact_threshold_m: float = 0.025,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = []
    masks_by_frame: dict[int, np.ndarray] = {}
    config = YellowSegmentationConfig()
    for frame_idx in range(num_frames):
        rgb = read_rgb_frame(video_path, frame_idx)
        if backend != "color_yellow":
            raise ValueError(f"Unsupported backend in this CLI pass: {backend}")
        obs, mask = detect_yellow_object(rgb, frame_idx=frame_idx, object_id=object_id, config=config)
        obs = save_mask_and_crop(image_rgb=rgb, mask=mask, obs=obs, output_dir=output_dir, crop_padding_px=12)
        if obs.bbox_xywh is not None:
            depth = load_depth_frame(depth_dir, frame_idx)
            _points, stats = mask_depth_to_points(
                depth,
                mask,
                intrinsics=intrinsics,
                min_depth_m=0.15,
                max_depth_m=2.5,
                stride=4,
            )
            obs.depth_median_m = stats.depth_median_m
            obs.depth_valid_ratio = stats.depth_valid_ratio
        observations.append(obs)
        masks_by_frame[frame_idx] = mask
    observations = fill_missing_observations(observations, max_gap=config.gap_fill_frames)
    write_object_tracks_json(
        observations={object_id: observations},
        output_path=output_dir / "object_tracks.json",
        output_dir=output_dir,
        backend=backend,
        video_path=video_path,
        depth_dir=depth_dir,
    )
    write_object_tracks_csv({object_id: observations}, output_dir / "object_tracks.csv")

    semantic = {"schema_version": "1.0", "provider": "none", "object_id": object_id, "status": "skipped"}
    if not no_qwen:
        key_frames = select_key_frames(observations, max_frames=4)
        crop_paths = [obs.crop_path for obs in observations if obs.frame_idx in key_frames and obs.crop_path is not None]
        semantic = call_qwen_semantics(crop_paths=crop_paths, model=qwen_model)
        semantic.update({"schema_version": "1.0", "object_id": object_id, "evidence_frames": key_frames})
    write_json_payload(semantic, output_dir / "semantic_annotations.json")

    contact_payload = {
        "schema_version": "1.0",
        "object_id": object_id,
        "distance_threshold_m": contact_threshold_m,
        "hand_tracks": str(hand_tracks_path) if hand_tracks_path else None,
        "hand_joints_npz": str(hand_joints_npz) if hand_joints_npz else None,
        "frames": [
            {
                "frame_idx": obs.frame_idx,
                "left_contact": False,
                "right_contact": False,
                "left_min_distance_m": None,
                "right_min_distance_m": None,
                "contact_points": [],
                "status": "hand_joints_npz_missing" if hand_joints_npz is None else "contact_disabled_until_task_8",
            }
            for obs in observations
        ],
    }
    write_json_payload(contact_payload, output_dir / "contact_annotations.json")

    if save_rrd:
        rr.init("RoboEgo Object Annotation", spawn=False)
        rr.save(str(output_dir / "object_annotation.rrd"))
        for obs in observations:
            rr.set_time("frame", sequence=obs.frame_idx)
            rgb = read_rgb_frame(video_path, obs.frame_idx)
            overlay = draw_object_bbox_overlay(rgb, obs, label=object_id)
            log_object_overlay("camera_opencv/camera/left/rgb_object_bbox_overlay", overlay)
    return {"object_id": object_id, "frame_count": num_frames, "output_dir": str(output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment and annotate manipulated objects in ego video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--intrinsics_json", default="tmp/baai_rectified_intrinsics.json")
    parser.add_argument("--hand_tracks", default="")
    parser.add_argument("--hand_joints_npz", default="")
    parser.add_argument("--backend", default="color_yellow", choices=["color_yellow"])
    parser.add_argument("--object_id", default="yellow_object_0")
    parser.add_argument("--num_frames", type=int, default=900)
    parser.add_argument("--no_qwen", action="store_true")
    parser.add_argument("--qwen_model", default="qwen-vl-plus")
    parser.add_argument("--no_save_rrd", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intrinsics_data = json.loads(Path(args.intrinsics_json).read_text())
    intrinsics = intrinsics_data.get("left", intrinsics_data)
    result = run_annotation(
        video_path=Path(args.video),
        depth_dir=Path(args.depth_dir),
        output_dir=Path(args.output_dir),
        intrinsics=intrinsics,
        hand_tracks_path=Path(args.hand_tracks) if args.hand_tracks else None,
        hand_joints_npz=Path(args.hand_joints_npz) if args.hand_joints_npz else None,
        backend=args.backend,
        object_id=args.object_id,
        num_frames=args.num_frames,
        no_qwen=args.no_qwen,
        save_rrd=not args.no_save_rrd,
        qwen_model=args.qwen_model,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run CLI smoke test**

Run:

```bash
pytest tests/objects/test_object_pipeline_cli.py -q
```

Expected: `1 passed`.

- [ ] **Step 7: Run BAAI offline command**

Run:

```bash
python scripts/object/run_object_annotation.py \
  --video data/baai/videos/left_rectified.mp4 \
  --depth_dir tmp/baai_foundation_depth_continuous_900 \
  --intrinsics_json tmp/baai_rectified_intrinsics.json \
  --hand_tracks tmp/baai_ego_hand_tracking_900_identity_gate/ego_hand_tracks.json \
  --hand_joints_npz tmp/baai_hand_joints21_identity_bbox_edge_gate/joints_21_camera_opencv.npz \
  --output_dir tmp/baai_object_annotation_900 \
  --backend color_yellow \
  --object_id yellow_object_0 \
  --num_frames 900 \
  --no_qwen
```

Expected: prints JSON with `"frame_count": 900`; `tmp/baai_object_annotation_900/object_tracks.json`, `contact_annotations.json`, `semantic_annotations.json`, and `object_annotation.rrd` exist.

- [ ] **Step 8: Update README**

Append to `README.md` after the standard flow:

```markdown
## Object Annotation

BAAI yellow-object segmentation, offline semantic status, contact annotations, and Rerun review:

```bash
python scripts/object/run_object_annotation.py \
  --video data/baai/videos/left_rectified.mp4 \
  --depth_dir tmp/baai_foundation_depth_continuous_900 \
  --intrinsics_json tmp/baai_rectified_intrinsics.json \
  --hand_tracks tmp/baai_ego_hand_tracking_900_identity_gate/ego_hand_tracks.json \
  --hand_joints_npz tmp/baai_hand_joints21_identity_bbox_edge_gate/joints_21_camera_opencv.npz \
  --output_dir tmp/baai_object_annotation_900 \
  --backend color_yellow \
  --object_id yellow_object_0 \
  --num_frames 900 \
  --no_qwen

rerun tmp/baai_object_annotation_900/object_annotation.rrd
```

The RGB review layer is `camera_opencv/camera/left/rgb_object_bbox_overlay`.
```

- [ ] **Step 9: Commit**

```bash
git add roboego_hand_vis/objects/rerun_export.py roboego_hand_vis/objects/annotation_io.py scripts/object/run_object_annotation.py tests/objects/test_object_pipeline_cli.py README.md
git commit -m "feat: add object annotation pipeline cli"
```

## Task 8: Replace Contact Disabled State With Hand Joints Geometry Contact

**Files:**
- Modify: `scripts/object/run_object_annotation.py`
- Modify: `roboego_hand_vis/objects/contact.py`
- Test: `tests/objects/test_object_pipeline_contact.py`

- [ ] **Step 1: Write failing test for hand-joints contact integration**

Create `tests/objects/test_object_pipeline_contact.py`:

```python
from pathlib import Path

import numpy as np

from roboego_hand_vis.objects.contact import contact_payload_from_joints_npz_and_object_points


def test_contact_payload_from_joints_npz_and_object_points_marks_left_contact(tmp_path: Path):
    left = np.ones((1, 21, 3), dtype=np.float32)
    right = np.ones((1, 21, 3), dtype=np.float32)
    left[:, :, 2] = 0.2
    right[:, :, 2] = 0.2
    left[0, 8] = [0.01, 0.0, 0.0]
    npz_path = tmp_path / "joints_21_camera_opencv.npz"
    np.savez(
        npz_path,
        left_joints_21_camera_opencv=left,
        right_joints_21_camera_opencv=right,
        left_valid=np.array([True]),
        right_valid=np.array([True]),
    )
    object_points = {0: np.array([[0.0, 0.0, 0.0]], dtype=np.float32)}
    payload = contact_payload_from_joints_npz_and_object_points(
        object_id="yellow_object_0",
        hand_joints_npz=npz_path,
        object_points_by_frame=object_points,
        threshold_m=0.025,
    )
    frame = payload["frames"][0]
    assert frame["left_contact"] is True
    assert frame["right_contact"] is False
    assert "left.index_tip" in frame["contact_points"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/objects/test_object_pipeline_contact.py -q
```

Expected: import failure for `contact_payload_from_joints_npz_and_object_points`.

- [ ] **Step 3: Implement contact payload helper**

Append to `roboego_hand_vis/objects/contact.py`:

```python
from pathlib import Path


def _side_contact_row(
    *,
    side_name: str,
    joints: np.ndarray | None,
    valid: bool,
    object_points: np.ndarray,
    threshold_m: float,
) -> tuple[bool, float | None, list[str], str]:
    if joints is None or not valid:
        return False, None, [], f"{side_name}_hand_missing"
    tips = extract_fingertips_from_joints21(joints)
    result = compute_fingertip_contact(tips, object_points, threshold_m=threshold_m)
    return (
        result.contact,
        result.min_distance_m,
        [f"{side_name}.{name}" for name in result.contact_points],
        "ok",
    )


def contact_payload_from_joints_npz_and_object_points(
    *,
    object_id: str,
    hand_joints_npz: Path,
    object_points_by_frame: dict[int, np.ndarray],
    threshold_m: float,
) -> dict:
    data = np.load(hand_joints_npz)
    left = data["left_joints_21_camera_opencv"].astype(np.float32)
    right = data["right_joints_21_camera_opencv"].astype(np.float32)
    left_valid = data["left_valid"].astype(bool)
    right_valid = data["right_valid"].astype(bool)
    n = max(len(left), len(right), max(object_points_by_frame.keys(), default=-1) + 1)
    frames = []
    for frame_idx in range(n):
        points = object_points_by_frame.get(frame_idx, np.zeros((0, 3), dtype=np.float32))
        left_contact, left_min, left_points, left_status = _side_contact_row(
            side_name="left",
            joints=left[frame_idx] if frame_idx < len(left) else None,
            valid=bool(left_valid[frame_idx]) if frame_idx < len(left_valid) else False,
            object_points=points,
            threshold_m=threshold_m,
        )
        right_contact, right_min, right_points, right_status = _side_contact_row(
            side_name="right",
            joints=right[frame_idx] if frame_idx < len(right) else None,
            valid=bool(right_valid[frame_idx]) if frame_idx < len(right_valid) else False,
            object_points=points,
            threshold_m=threshold_m,
        )
        row = {
            "frame_idx": frame_idx,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "left_min_distance_m": left_min,
            "right_min_distance_m": right_min,
            "contact_points": left_points + right_points,
            "status": "ok" if left_status == "ok" and right_status == "ok" else ",".join(
                status for status in [left_status, right_status] if status != "ok"
            ),
        }
        frames.append(row)
    return {
        "schema_version": "1.0",
        "object_id": object_id,
        "distance_threshold_m": threshold_m,
        "hand_joints_npz": str(hand_joints_npz),
        "frames": frames,
    }
```

- [ ] **Step 4: Wire CLI to compute object points per frame and contact payload**

In `scripts/object/run_object_annotation.py`, import:

```python
from roboego_hand_vis.objects.contact import contact_payload_from_joints_npz_and_object_points
```

Use the existing `hand_joints_npz` parameter from Task 7:

```python
hand_joints_npz: Path | None,
```

Inside `run_annotation`, create:

```python
object_points_by_frame: dict[int, np.ndarray] = {}
```

When `mask_depth_to_points` returns points:

```python
object_points_by_frame[frame_idx] = points
```

Replace the contact-disabled `contact_payload = {...}` block with:

```python
if hand_joints_npz is not None and hand_joints_npz.exists():
    contact_payload = contact_payload_from_joints_npz_and_object_points(
        object_id=object_id,
        hand_joints_npz=hand_joints_npz,
        object_points_by_frame=object_points_by_frame,
        threshold_m=contact_threshold_m,
    )
else:
    contact_payload = {
        "schema_version": "1.0",
        "object_id": object_id,
        "distance_threshold_m": contact_threshold_m,
        "hand_joints_npz": str(hand_joints_npz) if hand_joints_npz else None,
        "hand_tracks": str(hand_tracks_path) if hand_tracks_path else None,
        "frames": [
            {
                "frame_idx": obs.frame_idx,
                "left_contact": False,
                "right_contact": False,
                "left_min_distance_m": None,
                "right_min_distance_m": None,
                "contact_points": [],
                "status": "hand_joints_npz_missing",
            }
            for obs in observations
        ],
    }
```

Keep the Task 7 CLI argument:

```python
parser.add_argument("--hand_joints_npz", default="")
```

Keep passing it to `run_annotation`:

```python
hand_joints_npz=Path(args.hand_joints_npz) if args.hand_joints_npz else None,
```

- [ ] **Step 5: Run contact tests**

Run:

```bash
pytest tests/objects/test_contact.py tests/objects/test_object_pipeline_contact.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add roboego_hand_vis/objects/contact.py scripts/object/run_object_annotation.py tests/objects/test_object_pipeline_contact.py
git commit -m "feat: compute hand object contact annotations"
```

## Task 9: Final Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run full object test suite**

Run:

```bash
pytest tests/objects -q
```

Expected: all object tests pass.

- [ ] **Step 2: Run targeted existing tests**

Run:

```bash
pytest tests/depth/test_depth_pipeline.py tests/egohand/test_tracking_modules.py tests/egohand/test_track_fill_smooth_rerun_layers.py -q
```

Expected: selected existing tests pass.

- [ ] **Step 3: Run BAAI offline object annotation**

Run:

```bash
python scripts/object/run_object_annotation.py \
  --video data/baai/videos/left_rectified.mp4 \
  --depth_dir tmp/baai_foundation_depth_continuous_900 \
  --intrinsics_json tmp/baai_rectified_intrinsics.json \
  --hand_tracks tmp/baai_ego_hand_tracking_900_identity_gate/ego_hand_tracks.json \
  --hand_joints_npz tmp/baai_hand_joints21_identity_bbox_edge_gate/joints_21_camera_opencv.npz \
  --output_dir tmp/baai_object_annotation_900 \
  --backend color_yellow \
  --object_id yellow_object_0 \
  --num_frames 900 \
  --no_qwen
```

Expected:

```text
tmp/baai_object_annotation_900/object_tracks.json
tmp/baai_object_annotation_900/object_tracks.csv
tmp/baai_object_annotation_900/contact_annotations.json
tmp/baai_object_annotation_900/semantic_annotations.json
tmp/baai_object_annotation_900/object_annotation.rrd
```

- [ ] **Step 4: Inspect output counts**

Run:

```bash
python - <<'PY'
import json
root='tmp/baai_object_annotation_900'
tracks=json.load(open(f'{root}/object_tracks.json'))
frames=tracks['objects']['yellow_object_0']['frames']
contacts=json.load(open(f'{root}/contact_annotations.json'))['frames']
print(len(frames), sum(1 for f in frames if f['bbox_xywh'] is not None))
print(len(contacts), sum(1 for f in contacts if f['left_contact'] or f['right_contact']))
PY
```

Expected: first line starts with `900`; second line starts with `900`.

- [ ] **Step 5: Confirm implementation worktree is clean after task commits**

Run:

```bash
git status --short roboego_hand_vis/objects scripts/object tests/objects README.md
```

Expected: no output. If output appears, return to the task that introduced those files, complete its test/commit step, then rerun this status command.
