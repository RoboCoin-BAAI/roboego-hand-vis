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

## OpenVINS VIO

VIO is currently a standalone offline adapter, not a node in the Kedro hand
pipeline. It replays timestamped raw stereo images and IMU measurements into
OpenVINS and records both the IMU pose and the left-camera pose. It does not use
Qwen, other VLM credentials, depth, HaMeR, or SAM3.

### Inputs

The VIO session must contain this subset of the raw session layout:

```text
session/
├── videos/
│   ├── left.mp4
│   └── right.mp4
├── timestamps/
│   ├── left_timestamps.txt
│   └── right_timestamps.txt
├── imu/imu0.csv
└── calibrations/
    ├── camera_imu-camchain-imucam.yaml
    └── camera_imu-imu.yaml
```

Use the raw, unrectified camera videos. The runner loads the calibrated
equidistant camera model directly. `left/right` are OpenVINS `cam0/cam1`;
`bleft/bright` and audio may remain in a complete session but are not consumed
by the current two-camera VIO adapter.

Before estimation, the adapter validates video dimensions against calibration,
matches stereo frames by timestamp, removes repeated timestamps, checks IMU
coverage and gaps, converts Kalibr `T_cam_imu` to OpenVINS `T_imu_cam`, and
writes the OpenCV-YAML configuration used by OpenVINS.

### Dependencies

The estimator process requires:

```text
Ubuntu 22.04 / ROS 2 Humble       /opt/ros/humble/setup.bash
OpenVINS ROS 2 build              <OPENVINS_ROOT>/install_ros2/setup.bash
OpenVINS packages                 ov_core, ov_init, ov_msckf
Native build dependencies         Eigen3, Boost, Ceres
System Python                     rclpy, sensor_msgs, geometry_msgs,
                                  numpy, OpenCV, PyYAML
```

The runner intentionally uses `/usr/bin/python3` for ROS Python compatibility.
The project environment is used for Rerun export and visualization. A standard
OpenVINS build is:

```bash
sudo apt-get install libeigen3-dev libboost-all-dev libceres-dev \
  python3-numpy python3-opencv python3-yaml

cd /home/user/open_vins
source /opt/ros/humble/setup.bash
colcon --log-base log_ros2 build \
  --build-base build_ros2 \
  --install-base install_ros2 \
  --packages-select ov_core ov_init ov_msckf \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_ARUCO_TAGS=OFF
```

This workstation already has OpenVINS at `/home/user/open_vins/install_ros2`.
Its user-local Ceres libraries are under
`/home/user/.local/openvins-deps/usr`; the runner adds that library directory
automatically. Verify the environment with:

```bash
source /opt/ros/humble/setup.bash
source /home/user/open_vins/install_ros2/setup.bash
/usr/bin/python3 -c 'import cv2, numpy, yaml, rclpy; import geometry_msgs, sensor_msgs'
ros2 pkg executables ov_msckf
```

Runner defaults can be changed with `OPENVINS_ROOT`, `OPENVINS_INSTALL`,
`OPENVINS_DEPS`, `SYSTEM_PYTHON`, and `ROS_DOMAIN_ID`. `ffmpeg` is optional and
is only needed for the camera/trajectory comparison MP4.

### Run The One-Minute Slice

The complete test slice at `06:49-07:49` is:

```text
/home/user/ego_data/session_20260804_224442_clip_0649_60s/session
```

Run it into a new output directory:

```bash
cd /home/user/roboego-hand-vis

scripts/vio/run_openvins_session.sh \
  --session /home/user/ego_data/session_20260804_224442_clip_0649_60s/session \
  --output tmp/openvins_clip_0649_60s \
  --rate 1.0
```

Useful options:

```text
--start_s SECONDS       Offset from the first synchronized stereo pair
--duration_s SECONDS    Process only this duration; omit for the whole session
--rate RATE             Dataset seconds replayed per wall-clock second
--no-downsample         Track at the raw 1920x1088 resolution
--openvins_root PATH    Use PATH/install_ros2 instead of the default build
```

The runner launches OpenVINS in an isolated ROS domain, replays images and IMU
in timestamp order, waits rather than bursting IMU samples when video decoding
falls behind, records the trajectory, and shuts down the ROS processes on exit.

### Outputs And Coordinates

```text
output/
├── input/
│   ├── stereo_pairs.csv
│   ├── imu_clean.csv
│   └── input_report.json
├── config/
├── trajectory/
│   ├── trajectory.jsonl
│   ├── trajectory.npz
│   └── run_report.json
├── run_manifest.json
├── openvins.log
├── openvins_timing.txt
├── openvins_state_estimate.txt
└── openvins_state_deviation.txt
```

`trajectory.npz` is the integration artifact. It contains absolute camera/IMU
timestamps, source frame indices, covariance, `T_world_imu`, and
`T_world_camera`. `T_world_camera` is the left camera (`cam0`) pose derived from
the estimated IMU pose and the calibrated rigid transform:

```text
T_world_camera0 = T_world_imu @ T_imu_camera0
p_world = T_world_camera0 @ p_camera0
```

The OpenVINS world origin is local and arbitrary. Translation is metric because
the estimator uses the IMU, but this is not a globally registered map and the
current adapter has no loop closure. Join the result to later stages by camera
timestamp, or by frame index when both artifacts come from the same session.

### Rerun Visualization

Create a compact Rerun recording with a sampled left-camera reference:

```bash
/home/user/miniconda3/envs/egoall/bin/python \
  scripts/vio/export_trajectory_rerun.py \
  --trajectory tmp/openvins_clip_0649_60s/trajectory/trajectory.npz \
  --output tmp/openvins_clip_0649_60s/trajectory/trajectory_with_left.rrd \
  --camera_video /home/user/ego_data/session_20260804_224442_clip_0649_60s/session/videos/left.mp4 \
  --camera_timestamps /home/user/ego_data/session_20260804_224442_clip_0649_60s/session/timestamps/left_timestamps.txt \
  --camera_sample_fps 1 \
  --camera_max_width 640 \
  --jpeg_quality 70 \
  --pose_sample_fps 5 \
  --trajectory_chunk_seconds 5

/home/user/miniconda3/envs/egoall/bin/rerun \
  --memory-limit 2GB \
  tmp/openvins_clip_0649_60s/trajectory/trajectory_with_left.rrd
```

The trajectory appears incrementally as the timeline advances. The bright
yellow point is the current left-camera position. The full-rate trajectory
remains in `trajectory.npz`; Rerun sampling only affects visualization.

The current adapter is an offline reference implementation: decoding MP4 and
publishing ROS messages can be slower than the estimator itself. It is not yet
fused into the hand/depth outputs or the Kedro default pipeline. See
[`docs/pipeline/openvins_vio.md`](docs/pipeline/openvins_vio.md) for validation,
performance, long-recording visualization, and full-session details.

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
egohand      HaMeR, weak hand filtering, 21-joint pose gate, depth anchoring, final hand result
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
export DEPTH_DIR=$HANDWIN_DIR/depth/fast_foundation

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
  --params hamer_windows.enabled=True,hamer_windows.windows_json=$HAND_WINDOWS,hamer_windows.segments_root=$SEGMENTS_ROOT,hamer_windows.output_subdir=hand/hamer_raw,hamer_windows.hamer_root=/home/liuyou/Documents/hamer,hamer_windows.stage1_actions=$STAGE1,hamer_windows.hamer_python=/home/liuyou/miniconda3/envs/hamer/bin/python,hamer_windows.body_detector=regnety,hamer_windows.batch_size=1,hamer_windows.wearer_roi_x_margin_ratio=0.0,hamer_windows.wearer_roi_y_margin_ratio=0.10,hamer_windows.save_mesh=True,hamer_windows.refine_mode=strict_ego_audit,hamer_windows.only_window_ids=handwin_0000,hamer_windows.overwrite=True
```

Then set:

```bash
export HAMER_RUN=$HANDWIN_DIR/hand/hamer_raw
export REFINED_JSON=$HAMER_RUN/ego_hand_candidates_refined.json
export FILTER_DIR=$HANDWIN_DIR/hand/filtered
export JOINTS_GATE_DIR=$HANDWIN_DIR/hand/joints21_pose_gate
export ANCHORED_DIR=$HANDWIN_DIR/hand/joints21_depth_anchored
export FINAL_DIR=$HANDWIN_DIR/hand/final_hand_result
```

Weak hand filter:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes hand_filter \
  --params hand_filter.enabled=True,hand_filter.manifest=$MANIFEST,hand_filter.refined_json=$REFINED_JSON,hand_filter.output_dir=$FILTER_DIR,hand_filter.start=0,hand_filter.num_frames=1980,hand_filter.max_bbox_jump_px=360,hand_filter.max_bbox_area_change_ratio=8.0,hand_filter.max_mano_pose_jump=0,hand_filter.max_mesh_centroid_jump=0,hand_filter.max_mesh_vertex_rmse=0,hand_filter.local_window_radius=2,hand_filter.save_vis=False
```

21-joint pose gate:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes joints21_pose_gate \
  --params joints21_pose_gate.enabled=True,joints21_pose_gate.final_tracks_json=$FILTER_DIR/filtered_hand_tracks.json,joints21_pose_gate.output_dir=$JOINTS_GATE_DIR,joints21_pose_gate.start=0,joints21_pose_gate.num_frames=1980,joints21_pose_gate.drop_fingertips_below_wrist=True,joints21_pose_gate.image_height=1050,joints21_pose_gate.max_suspicious_wrist_y_ratio=0.55,joints21_pose_gate.min_fingertip_below_wrist_ratio=1.0,joints21_pose_gate.min_fingertip_below_wrist_delta=0.02
```

Depth-anchored 21 joints:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes depth_anchored_joints21 \
  --params depth_anchored_joints21.enabled=True,depth_anchored_joints21.manifest=$MANIFEST,depth_anchored_joints21.candidates_json=$JOINTS_GATE_DIR/final_hand_candidates_gated.json,depth_anchored_joints21.depth_dir=$DEPTH_DIR,depth_anchored_joints21.output_dir=$ANCHORED_DIR,depth_anchored_joints21.start=0,depth_anchored_joints21.num_frames=1980,depth_anchored_joints21.anchor_mode=wrist_palm,depth_anchored_joints21.max_depth_m=2.5,depth_anchored_joints21.overlay_frames=auto
```

Final hand result:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes final_hand_result \
  --params final_hand_result.enabled=True,final_hand_result.tracks_json=$JOINTS_GATE_DIR/final_hand_candidates_gated.json,final_hand_result.output_dir=$FINAL_DIR,final_hand_result.depth_anchored_json=$ANCHORED_DIR/joints21_depth_anchored.json,final_hand_result.depth_anchored_npz=$ANCHORED_DIR/joints21_depth_anchored.npz,final_hand_result.depth_anchored_summary_json=$ANCHORED_DIR/joints21_depth_anchored_summary.json,final_hand_result.copy_meshes=True
```

Final trajectory polish:

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
/home/liuyou/miniconda3/envs/ego/bin/kedro run \
  --pipelines egohand \
  --nodes final_hand_trajectory_polish \
  --params final_hand_trajectory_polish.enabled=True,final_hand_trajectory_polish.joints_json=$FINAL_DIR/joints21_depth_anchored.json,final_hand_trajectory_polish.output_dir=$FINAL_DIR,final_hand_trajectory_polish.manifest=$MANIFEST,final_hand_trajectory_polish.max_interp_gap=10,final_hand_trajectory_polish.wrist_jump_m=0.0,final_hand_trajectory_polish.max_local_wrist_deviation_m=0.0,final_hand_trajectory_polish.local_window_radius=5,final_hand_trajectory_polish.smooth_window=9,final_hand_trajectory_polish.smooth_relative_fingers=True,final_hand_trajectory_polish.write_preview=True
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
$HANDWIN_DIR/objects/sam3_object_tracks/
$HANDWIN_DIR/objects/sam3_object_tracks/combined/
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
hand_filter
joints21_pose_gate
depth_anchored_joints21
final_hand_result
final_hand_trajectory_polish
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
