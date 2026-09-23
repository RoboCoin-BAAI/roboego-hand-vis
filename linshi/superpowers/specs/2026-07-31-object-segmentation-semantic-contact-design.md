# Object Segmentation, Semantic Annotation, And Contact Design

## Goal

Add an object pipeline step to `roboego-hand-vis` that can:

1. Segment the manipulated object in BAAI-style ego stereo video.
2. Produce per-frame object masks, bounding boxes, crops, and depth point samples.
3. Call Qwen vision through a configurable API to assign structured semantic labels.
4. Estimate hand-object contact from depth and existing hand tracking geometry.
5. Export annotations and Rerun visualization layers for review.

The first implementation targets the BAAI yellow object sequence with a lightweight color backend. The API boundaries should allow a later SAM3 backend without changing downstream semantic/contact code.

## Non-Goals

- Do not implement full Do-as-I-Do object 6DoF tracking in this step.
- Do not generate a SAM3D-completed canonical mesh in this step.
- Do not ask Qwen to decide contact. Contact is a geometry/depth computation.
- Do not require Qwen for every frame. Semantic labeling should run on selected key crops and propagate to the object track.

## Inputs

Required:

- `dataset_manifest.json`, or explicit `--video`.
- Existing FoundationStereo depth directory, with files such as `frame_000000_foundation_depth_m.npy`.
- Camera intrinsics from the manifest or alignment report.

For contact:

- Existing hand tracking JSON, such as `tmp/baai_ego_hand_tracking_900_identity_gate/ego_hand_tracks.json`.
- Hand mesh files referenced by the hand tracking observations.

For Qwen semantic annotation:

- `DASHSCOPE_API_KEY` or an explicit API key environment variable.
- Configurable `--qwen_base_url`, defaulting to the DashScope OpenAI-compatible endpoint.
- Configurable `--qwen_model`, defaulting to a Qwen vision-capable model chosen by the user/environment.

## Outputs

Create an output directory such as:

```text
tmp/<dataset>_object_annotation_<range>/
```

The directory contains:

```text
object_tracks.json
object_tracks.csv
semantic_annotations.json
contact_annotations.json
crops/
  frame_000000_yellow_object_0.png
masks/
  frame_000000_yellow_object_0.png
debug/
  bbox_contact_preview.jpg
object_annotation.rrd
```

### `object_tracks.json`

Per-frame segmentation and depth summary:

```json
{
  "schema_version": "1.0",
  "backend": "color_yellow",
  "objects": {
    "yellow_object_0": {
      "frames": [
        {
          "frame_idx": 0,
          "bbox_xywh": [762, 580, 142, 176],
          "mask_path": "masks/frame_000000_yellow_object_0.png",
          "crop_path": "crops/frame_000000_yellow_object_0.png",
          "mask_area_px": 20140,
          "depth_median_m": 0.354,
          "depth_valid_ratio": 0.98,
          "source": "yellow_hsv_component"
        }
      ]
    }
  }
}
```

### `semantic_annotations.json`

Object-level semantics assigned from selected key crops:

```json
{
  "schema_version": "1.0",
  "provider": "qwen",
  "object_id": "yellow_object_0",
  "category": "box",
  "color": "yellow",
  "shape": "rectangular box",
  "material": "paper or cardboard",
  "visual_description": "A yellow box-like object with a smiley face graphic.",
  "confidence": 0.82,
  "evidence_frames": [0, 120, 840]
}
```

### `contact_annotations.json`

Frame-level contact computed from geometry:

```json
{
  "schema_version": "1.0",
  "object_id": "yellow_object_0",
  "distance_threshold_m": 0.025,
  "frames": [
    {
      "frame_idx": 120,
      "left_contact": true,
      "right_contact": false,
      "left_min_distance_m": 0.014,
      "right_min_distance_m": 0.061,
      "contact_points": ["left.index_tip", "left.thumb_tip"]
    }
  ]
}
```

## Architecture

Add a new package area:

```text
roboego_hand_vis/objects/
  __init__.py
  segmentation.py
  color_yellow.py
  sam3_backend.py
  depth_geometry.py
  semantic_qwen.py
  contact.py
  annotation_io.py
  rerun_export.py
```

Add CLIs:

```text
scripts/object/run_object_annotation.py
scripts/object/export_object_rerun.py
```

### Segmentation Interface

All backends return the same structure:

```python
ObjectObservation(
    frame_idx: int,
    object_id: str,
    bbox_xywh: list[int] | None,
    mask: np.ndarray | None,
    source: str,
)
```

The initial backend is `color_yellow`, using HSV thresholding, ROI filtering, connected components, morphology, and short-gap interpolation. It is deterministic and testable without model weights.

The first `sam3` backend implementation should support loading an existing SAM3-style masks directory. A later implementation can call a SAM3 runner directly. Both paths must emit the same `ObjectObservation` objects as `color_yellow`.

### Depth Geometry

For each object mask:

- Load the matching depth frame.
- Filter invalid depth and extreme depth outliers.
- Back-project mask pixels into the left camera OpenCV frame.
- Store summary stats and optionally sampled points.

This reuses the same camera convention as hand tracking:

```text
camera_opencv, unit=meter
```

### Semantic Annotation

Select key crops rather than every frame:

- First valid frame.
- Largest mask area frame.
- Several evenly spaced valid frames.
- Optional frames near first/last contact once contact is available.

Call Qwen with a prompt that asks for strict JSON:

```json
{
  "category": "string",
  "color": "string",
  "shape": "string",
  "material": "string",
  "visual_description": "string",
  "confidence": 0.0
}
```

The implementation should:

- Read API credentials from environment.
- Avoid logging API keys.
- Cache model responses in `semantic_annotations.json`.
- Support `--no_qwen` to run segmentation/contact offline.
- Validate and repair JSON responses conservatively.

### Contact Computation

Contact is computed from object depth points and hand geometry:

1. For each frame, get object point cloud from mask + depth.
2. For each hand, load final smoothed hand mesh if available.
3. Also compute 21 joints or fingertip points.
4. Measure nearest-neighbor distance from fingertips to object points.
5. Optionally measure sampled hand mesh vertices to object points for palm/side contact.
6. Mark contact when distance is below a threshold, default `0.025m`.

The first pass should prioritize fingertip contact because it is cheap and easier to debug:

```text
thumb_tip, index_tip, middle_tip, ring_tip, little_tip
```

If the mesh is available, add mesh-level min distance as extra evidence, not as the only criterion.

## Rerun Export

Add review layers:

```text
camera_opencv/camera/left/rgb_object_bbox_overlay
camera_opencv/camera/left/rgb_object_contact_overlay
camera_opencv/objects/<object_id>/depth_points
camera_opencv/objects/<object_id>/contact_points
```

For RGB review, draw boxes and labels directly into an image layer. This avoids Rerun `Boxes2D` path ambiguity where boxes can appear in the 3D/point-cloud view instead of as image overlays.

Rerun text or labels should include:

- object id
- semantic category/color
- contact side
- min distance

## CLI Shape

Example:

```bash
python scripts/object/run_object_annotation.py \
  --manifest tmp/baai_dataset_manifest/dataset_manifest.json \
  --video data/baai/videos/left_rectified.mp4 \
  --depth_dir tmp/baai_foundation_depth_continuous_900 \
  --hand_tracks tmp/baai_ego_hand_tracking_900_identity_gate/ego_hand_tracks.json \
  --output_dir tmp/baai_object_annotation_900 \
  --backend color_yellow \
  --semantic_provider qwen \
  --qwen_model qwen-vl-plus \
  --num_frames 900
```

Offline/debug:

```bash
python scripts/object/run_object_annotation.py \
  --manifest tmp/baai_dataset_manifest/dataset_manifest.json \
  --depth_dir tmp/baai_foundation_depth_continuous_900 \
  --hand_tracks tmp/baai_ego_hand_tracking_900_identity_gate/ego_hand_tracks.json \
  --output_dir tmp/baai_object_annotation_900 \
  --backend color_yellow \
  --no_qwen
```

## Error Handling

- Missing depth frame: keep segmentation output, mark contact as unavailable for that frame.
- Missing hand observation: mark that hand as `contact_available=false`.
- Missing API key: skip Qwen when `--no_qwen` is set; otherwise fail with a clear message.
- Invalid Qwen JSON: save raw response, set semantic status to `parse_failed`, and keep the run usable.
- Empty object mask: interpolate short gaps, otherwise mark object absent.

## Testing

Unit tests:

- Color segmentation returns a bbox for synthetic yellow rectangles.
- Gap interpolation fills short missing spans and leaves long spans marked missing.
- Depth back-projection returns expected 3D points for a known intrinsics/depth fixture.
- Contact thresholding marks true/false contact for synthetic fingertip/object points.
- Qwen response parser accepts strict JSON and handles malformed JSON without crashing.

Integration smoke test:

- Run `--backend color_yellow --no_qwen` on a tiny synthetic or sampled video fixture.
- Confirm `object_tracks.json`, `contact_annotations.json`, and Rerun output paths are written.

Manual verification:

- Open the Rerun output and inspect `rgb_object_bbox_overlay` and `rgb_object_contact_overlay`.
- Spot-check BAAI frames with occlusion and contact.

## Rollout

1. Implement deterministic object segmentation and annotation IO.
2. Add Qwen semantic annotation with caching and `--no_qwen`.
3. Add depth/fingertip contact computation.
4. Add Rerun overlays.
5. Add SAM3 backend interface stub and documentation for future wiring.
