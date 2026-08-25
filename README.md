# RoboEgo Hand Vis

RoboEgo Hand Vis is a Kedro-based processing pipeline for head-mounted stereo
ego-hand data. The current architecture is organized around stage pipelines
instead of one-off shell scripts:

```text
quality -> semantic -> datasets -> restoration -> depth -> egohand -> objects
```

The main use case is processing a raw headband session into selected hand-object
interaction windows with stereo videos, rectified dataset manifests, depth,
HaMeR hand meshes, filtered 21-joint hand tracks, and SAM3 object masks.

## Environment

The project is usually run from the `ego` environment:

```bash
cd /home/liuyou/Documents/roboego-hand-vis
source /home/liuyou/miniconda3/etc/profile.d/conda.sh
conda activate ego

pip install -e ".[dev]"
```

Expected local Python environments:

```text
ego      Main Kedro pipeline, quality checks, semantic glue, datasets, hand filtering
py312    Fast-FoundationStereo and SAM3 wrappers
hamer    HaMeR inference
realvdeblur  Optional RealVDeblur video restoration
```

External repositories used by specific nodes:

```text
/home/liuyou/Documents/RoboEgoPipe
/home/liuyou/Documents/Fast-FoundationStereo
/home/liuyou/Documents/FoundationStereo
/home/liuyou/Documents/hamer
/home/liuyou/Documents/do-as-i-do/sam3/sam3.pt
/home/liuyou/Documents/RealVDeblur
```

For Qwen annotation, set:

```bash
export DASHSCOPE_API_KEY=...
```

Kedro telemetry can be disabled for reproducible logs:

```bash
export KEDRO_DISABLE_TELEMETRY=1
```

## Pipeline Registry

Available Kedro pipelines:

```bash
PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python - <<'PY'
from roboego_hand_vis.pipeline_registry import register_pipelines
for name in sorted(register_pipelines()):
    print(name)
PY
```

Important pipelines:

```text
quality      Raw data integrity, Stage0 visual quality, stereo/depth quality
semantic     Qwen frame annotation, Stage1 actions, processing plan, hand windows
datasets     Segment export, rectification/manifest preparation, hand-window export
restoration  Optional RealVDeblur stereo restoration
depth        FoundationStereo and Fast-FoundationStereo
egohand      HaMeR, temporal filtering, final hand result, 21-joint filtering/depth anchoring
objects      SAM3 object segmentation for hand-object interaction windows
```

All runtime parameters live in:

```text
conf/base/parameters.yml
```

Most nodes are disabled by default. Enable only the node or stage you want with
`--params <section>.enabled=True,...`.

## Raw Headband Flow

Set paths for a raw session:

```bash
cd /home/liuyou/Documents/roboego-hand-vis

export SESSION=/home/liuyou/Documents/roboego-hand-vis/data/session_20260818_182102
export PLAN=$SESSION/annotation/processing_plan_left_v1.json
export HAND_WINDOWS=$SESSION/annotation/hand_estimation_windows_left_v1.json
export SEGMENTS_ROOT=$SESSION/segments
export STAGE1=$SESSION/annotation/qwen_frame_annotations_left_v4.checkpoint.json
```

If shell variables are not preserved in your terminal, use absolute paths
directly in `--params`.

### 1. Raw Data Quality

Checks camera videos, timestamps, timestamp alignment, video fps, IMU timestamps,
and IMU signal quality. Quaternion columns are ignored by default.

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines quality \
  --to-nodes raw_headband_quality_gate \
  --params raw_headband_quality.enabled=True,raw_headband_quality.session_dir=$SESSION,raw_headband_quality.output_root=$SESSION,raw_headband_quality.overwrite=True,raw_headband_quality_gate.strict=False
```

Output:

```text
$SESSION/annotation/quality_report.json
```

Key fields:

```text
timestamp_alignment
timestamp_duplicates
imu.signal_quality
video_infos
```

### 2. Stage0 Visual Quality

Finds valid visual intervals before sending frames to the VLM.

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines quality \
  --to-nodes pre_semantic_quality_gate \
  --params raw_headband_quality.enabled=True,raw_headband_quality.session_dir=$SESSION,raw_headband_quality.output_root=$SESSION,raw_headband_quality.overwrite=True,raw_headband_quality_gate.strict=False,pre_semantic_quality.enabled=True,pre_semantic_quality.video=$SESSION/videos/left.mp4,pre_semantic_quality.output_json=$SESSION/annotation/pre_semantic_quality_gate_left.json
```

Output:

```text
$SESSION/annotation/pre_semantic_quality_gate_left.json
```

### 3. Qwen Frame Annotation

Runs Qwen on Stage0 valid windows. The node writes batch outputs, chunks, a
checkpoint, and a final merged JSON. Remote API failures such as multimodal
download timeout are retried within the batch.

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --to-nodes qwen_frame_annotation \
  --params raw_headband_quality.enabled=True,raw_headband_quality.session_dir=$SESSION,raw_headband_quality.output_root=$SESSION,raw_headband_quality.overwrite=True,raw_headband_quality_gate.strict=False,pre_semantic_quality.enabled=True,pre_semantic_quality.video=$SESSION/videos/left.mp4,pre_semantic_quality.output_json=$SESSION/annotation/pre_semantic_quality_gate_left.json,qwen_frame_annotation.enabled=True,qwen_frame_annotation.video=$SESSION/videos/left.mp4,qwen_frame_annotation.output_json=$SESSION/annotation/qwen_frame_annotations_left_v4.json,qwen_frame_annotation.checkpoint_json=$SESSION/annotation/qwen_frame_annotations_left_v4.checkpoint.json,qwen_frame_annotation.raw_output_dir=$SESSION/annotation/qwen_frame_annotation_raw,qwen_frame_annotation.batch_output_dir=$SESSION/annotation/qwen_frame_annotation_batches,qwen_frame_annotation.chunk_output_dir=$SESSION/annotation/qwen_frame_annotation_chunks,qwen_frame_annotation.frames_dir=$SESSION/annotation/qwen_frame_annotation_frames,qwen_frame_annotation.prompt=docs/prompts/qwen_frame_annotation_v5_en.md,qwen_frame_annotation.fps=1,qwen_frame_annotation.max_width=640,qwen_frame_annotation.batch_size=4,qwen_frame_annotation.batches_per_chunk=5,qwen_frame_annotation.model=qwen3.6-flash,qwen_frame_annotation.timeout_s=180,qwen_frame_annotation.max_retries=5,qwen_frame_annotation.retry_sleep_s=3,qwen_frame_annotation.resume=True
```

Important outputs:

```text
$SESSION/annotation/qwen_frame_annotations_left_v4.checkpoint.json
$SESSION/annotation/qwen_frame_annotation_chunks/
$SESSION/annotation/qwen_frame_annotation_batches/
$SESSION/annotation/qwen_frame_annotation_raw/
```

### 4. Stage1 Actions

Extracts per-frame hand actions, builds text review chunks, and resolves action
word/object binding differences.

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines semantic \
  --from-nodes stage1_action_extraction_chunks \
  --to-nodes stage1_action_text_resolver_chunks \
  --params qwen_frame_annotation.enabled=False,stage1_action_chunks.enabled=True,stage1_action_chunks.qwen_chunk_dir=$SESSION/annotation/qwen_frame_annotation_chunks,stage1_action_chunks.action_chunk_dir=$SESSION/annotation/stage1_action_chunks,stage1_action_chunks.review_chunk_dir=$SESSION/annotation/stage1_action_text_review_chunks,stage1_action_chunks.resolved_chunk_dir=$SESSION/annotation/stage1_action_resolved_chunks,stage1_action_chunks.model=qwen-plus,stage1_action_chunks.overwrite=False
```

Output:

```text
$SESSION/annotation/stage1_action_resolved_chunks/
```

### 5. Processing Plan

Builds process/drop segments from Qwen annotations and resolved Stage1 actions.
`processing_plan.actions_json` can be either one JSON file or the resolved chunk
directory.

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines semantic \
  --nodes build_processing_plan \
  --params processing_plan.enabled=True,processing_plan.stage1_json=$SESSION/annotation/qwen_frame_annotations_left_v4.checkpoint.json,processing_plan.actions_json=$SESSION/annotation/stage1_action_resolved_chunks,processing_plan.output_json=$PLAN,processing_plan.min_process_segment_s=5,processing_plan.context_expand_s=2
```

Inspect:

```bash
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["PLAN"])
d = json.loads(p.read_text(encoding="utf-8"))
print("segments:", len(d["segments"]))
print("drop_segments:", len(d["drop_segments"]))
for s in d["segments"]:
    print(s["segment_id"], s["start"], s["end"], s["duration"], s["semantic_label"], s["actions"], s["objects"])
PY
```

### 6. Hand Estimation Windows

Builds focused hand windows from the processing plan. `stage1_actions` can also
be a resolved chunk directory.

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines semantic \
  --nodes build_hand_estimation_windows \
  --params hand_estimation_windows.enabled=True,hand_estimation_windows.stage1_actions=$SESSION/annotation/stage1_action_resolved_chunks,hand_estimation_windows.processing_plan=$PLAN,hand_estimation_windows.output_json=$HAND_WINDOWS,hand_estimation_windows.hand_window_mode=two,hand_estimation_windows.context_expand_s=2,hand_estimation_windows.merge_gap_s=2,hand_estimation_windows.min_window_s=5,hand_estimation_windows.min_priority=high,hand_estimation_windows.fps=30
```

Inspect:

```bash
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["HAND_WINDOWS"])
d = json.loads(p.read_text(encoding="utf-8"))
print("window_count:", d.get("window_count", len(d.get("windows", []))))
for w in d.get("windows", []):
    print(w["window_id"], w["start"], w["end"], w["duration"], w["window_type"], w["priority"], w["actions"], w["objects"])
PY
```

## Dataset and Hand-Window Preparation

Exports selected process segments, prepares rectified stereo data and manifests,
then exports hand-window clips.

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines datasets \
  --from-nodes export_selected_video_segments \
  --to-nodes export_hand_window_segments \
  --params export_selected_video_segments.enabled=True,export_selected_video_segments.session=$SESSION,export_selected_video_segments.processing_plan=$PLAN,export_selected_video_segments.output_dir=$SEGMENTS_ROOT,export_selected_video_segments.overwrite=True,prepare_selected_segments.enabled=True,prepare_selected_segments.session=$SESSION,prepare_selected_segments.segments_root=$SEGMENTS_ROOT,prepare_selected_segments.python=/home/liuyou/miniconda3/envs/ego/bin/python,prepare_selected_segments.roboegopipe_root=/home/liuyou/Documents/RoboEgoPipe,prepare_selected_segments.dataset_adapter=wumei01,prepare_selected_segments.model=equidistant,prepare_selected_segments.balance=0.0,prepare_selected_segments.width=1600,prepare_selected_segments.height=900,prepare_selected_segments.overwrite=True,export_hand_window_segments.enabled=True,export_hand_window_segments.segments_root=$SEGMENTS_ROOT,export_hand_window_segments.hand_windows_json=$HAND_WINDOWS,export_hand_window_segments.overwrite=True
```

Find the active hand window:

```bash
export HANDWIN_DIR=$(find "$SEGMENTS_ROOT" -path "*handwin_0000*" -type d | head -1)
export MANIFEST=$HANDWIN_DIR/dataset_manifest/dataset_manifest.json

echo "$HANDWIN_DIR"
echo "$MANIFEST"
```

## Stereo Quality and Depth

Stereo quality before depth:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines quality \
  --nodes stereo_summary_quality \
  --params stereo_summary_quality.enabled=True,stereo_summary_quality.manifest=$MANIFEST,stereo_summary_quality.output_dir=$HANDWIN_DIR/quality/stereo_summary

/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines quality \
  --nodes stereo_segment_quality \
  --params stereo_segment_quality.enabled=True,stereo_segment_quality.manifest=$MANIFEST,stereo_segment_quality.output_dir=$HANDWIN_DIR/quality/stereo_segments,stereo_segment_quality.sample_every_s=1.0,stereo_segment_quality.orb_every_s=5.0
```

Fast-FoundationStereo depth:

```bash
export DEPTH_DIR=$HANDWIN_DIR/fast_foundation_depth

/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines depth \
  --nodes fast_foundation_stereo_depth \
  --params fast_foundation_stereo_depth.enabled=True,fast_foundation_stereo_depth.manifest=$MANIFEST,fast_foundation_stereo_depth.output_dir=$DEPTH_DIR,fast_foundation_stereo_depth.start=0,fast_foundation_stereo_depth.num_frames=120,fast_foundation_stereo_depth.scale=0.5,fast_foundation_stereo_depth.valid_iters=4,fast_foundation_stereo_depth.max_disp=192,fast_foundation_stereo_depth.max_depth_m=4.0,fast_foundation_stereo_depth.overwrite=True
```

Start with `num_frames=120` to estimate speed. For a full window, use the
window frame count from `hand_estimation_windows_left_v1.json`.

## Ego-Hand Pipeline

Run HaMeR on the hand window:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes hamer_windows \
  --params hamer_windows.enabled=True,hamer_windows.windows_json=$HAND_WINDOWS,hamer_windows.segments_root=$SEGMENTS_ROOT,hamer_windows.output_subdir=hamer_windows_left_v1,hamer_windows.hamer_root=/home/liuyou/Documents/hamer,hamer_windows.stage1_actions=$STAGE1,hamer_windows.hamer_python=/home/liuyou/miniconda3/envs/hamer/bin/python,hamer_windows.body_detector=regnety,hamer_windows.batch_size=1,hamer_windows.wearer_roi_x_margin_ratio=0.0,hamer_windows.wearer_roi_y_margin_ratio=0.10,hamer_windows.save_mesh=True,hamer_windows.refine_mode=strict_ego_audit,hamer_windows.only_window_ids=handwin_0000,hamer_windows.overwrite=True
```

Then set:

```bash
export HAMER_RUN=$(find "$HANDWIN_DIR" -path "*hamer_windows_left_v1/handwin_*" -type d | head -1)
export REFINED_JSON=$HAMER_RUN/ego_hand_candidates_refined.json
export TEMPORAL_DIR=$HAMER_RUN/refined_temporal
export FINAL_DIR=$HAMER_RUN/final_hand_result
export JOINTS_GATE_DIR=$HAMER_RUN/joints21_pose_gate
export ANCHORED_DIR=$HAMER_RUN/joints21_depth_anchored
```

Temporal outlier removal:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes refined_temporal_gate \
  --params refined_temporal_gate.enabled=True,refined_temporal_gate.manifest=$MANIFEST,refined_temporal_gate.refined_json=$REFINED_JSON,refined_temporal_gate.output_dir=$TEMPORAL_DIR,refined_temporal_gate.start=0,refined_temporal_gate.num_frames=1980,refined_temporal_gate.max_bbox_jump_px=220,refined_temporal_gate.max_bbox_area_change_ratio=3.0,refined_temporal_gate.max_mano_pose_jump=1.6,refined_temporal_gate.max_mesh_centroid_jump=0.18,refined_temporal_gate.max_mesh_vertex_rmse=0.08,refined_temporal_gate.local_window_radius=2,refined_temporal_gate.save_vis=True
```

Final hand result:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes final_hand_result \
  --params final_hand_result.enabled=True,final_hand_result.tracks_json=$TEMPORAL_DIR/refined_temporal_gate_tracks.json,final_hand_result.output_dir=$FINAL_DIR,final_hand_result.copy_meshes=True,final_hand_result.copy_visuals=True
```

21-joint pose gate:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes joints21_pose_gate \
  --params joints21_pose_gate.enabled=True,joints21_pose_gate.final_tracks_json=$FINAL_DIR/final_hand_tracks.json,joints21_pose_gate.output_dir=$JOINTS_GATE_DIR,joints21_pose_gate.start=0,joints21_pose_gate.num_frames=1980
```

Depth-anchored 21 joints:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes depth_anchored_joints21 \
  --params depth_anchored_joints21.enabled=True,depth_anchored_joints21.manifest=$MANIFEST,depth_anchored_joints21.candidates_json=$JOINTS_GATE_DIR/final_hand_candidates_gated.json,depth_anchored_joints21.depth_dir=$DEPTH_DIR,depth_anchored_joints21.output_dir=$ANCHORED_DIR,depth_anchored_joints21.start=0,depth_anchored_joints21.num_frames=1980,depth_anchored_joints21.anchor_mode=wrist_palm,depth_anchored_joints21.max_depth_m=2.5,depth_anchored_joints21.overlay_frames=auto
```

## Object Segmentation

SAM3 runs only on hand-object interaction objects inferred from the Qwen/Stage1
annotations.

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines objects \
  --nodes sam3_hand_object_segmentation \
  --params sam3_hand_object_segmentation.enabled=True,sam3_hand_object_segmentation.hand_window_dir=$HAMER_RUN,sam3_hand_object_segmentation.stage1_actions=$STAGE1,sam3_hand_object_segmentation.windows_json=$HAND_WINDOWS,sam3_hand_object_segmentation.auto_from_stage1=True,sam3_hand_object_segmentation.sam3_python=/home/liuyou/miniconda3/envs/py312/bin/python,sam3_hand_object_segmentation.checkpoint=/home/liuyou/Documents/do-as-i-do/sam3/sam3.pt,sam3_hand_object_segmentation.fps=30
```

Output:

```text
$HAMER_RUN/sam3_object_tracks/
$HAMER_RUN/sam3_object_tracks/combined/
```

## Optional Nodes

RealVDeblur stereo restoration:

```bash
kedro run --pipelines restoration --nodes realvdeblur_stereo --params ...
```

Legacy mesh-depth alignment is still available but is no longer the recommended
main path:

```text
legacy_mesh_depth_alignment
legacy_track_fill_smooth
legacy_export_joints21
```

The recommended hand/depth path is:

```text
hamer_windows
refined_temporal_gate
final_hand_result
joints21_pose_gate
depth_anchored_joints21
```

## External Processed Datasets

For already-processed stereo datasets such as Ropedia, resolve a
`dataset_manifest.json` first:

```bash
PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python scripts/datasets/resolve_dataset.py \
  --dataset ropedia \
  --root /home/liuyou/Documents/ropedia_data \
  --output_dir tmp/ropedia_dataset_manifest
```

Then run the same quality, depth, egohand, and objects stages using the resolved
manifest.

## Development

Run tests:

```bash
PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python -m pytest -q
```

Run a subset:

```bash
PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python -m pytest tests/pipelines -q
```

Generated data, videos, model outputs, masks, meshes, and `.rrd` files should
not be committed. Keep large outputs under `data/`, `tmp/`, or external storage.
