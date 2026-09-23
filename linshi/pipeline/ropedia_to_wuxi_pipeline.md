# Ropedia / Wuxi Ego Hand Visualization Pipeline

本文档记录当前确定下来的 Ropedia 与 Wuxi 数据处理标准流程。目标是从双目视频得到：

- FoundationStereo 深度
- HaMeR / MANO 手部 mesh
- ego_left / ego_right 候选筛选、恢复、缺失补帧
- shape-preserved MANO + depth anchor 对齐
- finger anchor depth
- orientation-aware surface-to-skeleton compensation
- Rerun 可视化 RRD
- Ropedia 官方 3D hand annotation 对比评估

## 0. 总体约定

### 坐标系

```text
camera_opencv:
  x: right
  y: down
  z: forward
  unit: meter

  用于 depth point cloud、MANO mesh、joints、MPJPE 等数值计算。

camera_display:
  Rerun 观看用坐标。
  只用于可视化，不作为精度评估坐标。
```

### 数据约定

Ropedia 默认输入：

```text
left video:  /path/to/ropedia_data/stereo_left.mp4
right video: /path/to/ropedia_data/stereo_right.mp4
depth dir:   tmp/ropedia_foundation_depth_continuous_000000_000299
```

Wuxi 默认输入：

```text
left video:  data/wuxi/stereo_left.mp4
right video: data/wuxi/stereo_right.mp4
intrinsics:  data/wuxi/camera_intrinsics.json
depth dir:   tmp/wuxi_foundation_depth_continuous_000000_000299
```

Wuxi 必须使用自己的 `camera_intrinsics.json`。如果误用 Ropedia 默认内参，会导致点云尺度和 mesh 尺寸明显错误。

### 标准 manifest 字段

`dataset_manifest.json` 是所有数据源 adapter 的统一输出，核心结构如下：

```text
schema_version:
  manifest schema 版本。

dataset_name / sequence_name / root_dir:
  数据源名称、序列名称、数据根目录。

videos.left / videos.right:
  左右视频路径、width、height、fps、frame_count、duration_sec。

camera_model:
  camera type、是否 rectified、是否 undistorted、calibration source。

intrinsics.left / intrinsics.right:
  fx、fy、cx、cy、width、height、distortion_model、distortion_coeffs。

extrinsics:
  baseline_m、left_to_right、raw left/right/depth extrinsics。

timestamps:
  帧时间戳来源。没有独立时间戳时为 frame_index。

trajectory:
  tracking_path、head_pose_available、imu_available、hand_tracking_available。

annotations:
  官方标注路径和可用性。Ropedia 有 annotation_hdf5，Wuxi 当前为空。

reference_camera:
  当前固定为 left。后续 depth、point cloud、MANO mesh 都以 left camera 为参考。
```

### 环境

```bash
cd /path/to/roboego-hand-vis
source /path/to/conda/etc/profile.d/conda.sh
conda activate ego
```

HaMeR 代码和 MANO 模型默认位置：

```text
HAMER_ROOT=/path/to/hamer
MANO_MODEL_DIR=/path/to/hamer/_DATA/data/mano
```

FoundationStereo 默认位置：

```text
FOUNDATION_ROOT=/path/to/FoundationStereo
FOUNDATION_CKPT=/path/to/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
```

## 1. 一键入口

### 先生成标准 manifest

工程化流程的第一步是把不同数据源统一解析成 `dataset_manifest.json`。这一步不做质量判断，只回答：

```text
左眼视频在哪里？
右眼视频在哪里？
视频分辨率 / fps / 帧数是多少？
相机内参是什么？
左右相机 baseline / 外参是什么？
是否有 tracking / head pose / IMU？
是否有官方 annotation？
```

Wuxi：

```bash
python scripts/datasets/resolve_dataset.py \
  --dataset wuxi \
  --root data/wuxi \
  --output_dir tmp/wuxi_dataset_manifest
```

Ropedia：

```bash
python scripts/datasets/resolve_dataset.py \
  --dataset ropedia \
  --root /path/to/ropedia_data \
  --output_dir tmp/ropedia_dataset_manifest
```

Generic stereo：

```bash
python scripts/datasets/resolve_dataset.py \
  --dataset generic_stereo \
  --root /path/to/sequence \
  --left_video stereo_left.mp4 \
  --right_video stereo_right.mp4 \
  --intrinsics_json camera_intrinsics.json \
  --output_dir tmp/my_dataset_manifest
```

输出：

```text
tmp/<dataset>_dataset_manifest/
  dataset_manifest.json
  dataset_summary.txt
```

后续质检和模型流程应优先读取这个 manifest，而不是在每个脚本里重新手写 left/right video 和 intrinsics 路径。

### Dataset adapter 代码结构

不同数据源不要都写进 `manifest.py`。当前结构是：

```text
roboego_hand_vis/datasets/common.py
  公共 schema、ResolveConfig、视频 probe、相机参数解析、tracking/timestamps、summary。

roboego_hand_vis/datasets/wuxi.py
  Wuxi 数据源 adapter。

roboego_hand_vis/datasets/ropedia.py
  Ropedia 数据源 adapter。

roboego_hand_vis/datasets/generic_stereo.py
  通用双目数据源 adapter。

roboego_hand_vis/datasets/manifest.py
  只做 resolver 注册、统一入口 resolve_dataset_manifest、write_manifest。
```

新增数据源时，推荐做法是：

```text
1. 新增 roboego_hand_vis/datasets/<dataset_name>.py
2. 在里面实现 resolve(config: ResolveConfig) -> dict
3. 复用 common.py 里的 base_manifest / probe_video / intrinsics helpers
4. 在 manifest.py 的 RESOLVERS 里注册一次
5. 后续质检、深度、HaMeR、Rerun 流程只读取 dataset_manifest.json
```

也就是说，每个数据源可以有自己的加载规则，但最终都必须输出同一种 `dataset_manifest.json`。

## 2. Step 1: 数据质检

质检只读取 `dataset_manifest.json`，不再手写左右视频路径。这样 Wuxi、Ropedia、后续新数据源都走同一套检查逻辑。

### 快速抽样质检

用途：先快速判断这段双目视频是否大体适合进入 FoundationStereo 和 HaMeR。

```bash
python scripts/quality/check_stereo_video_quality.py \
  --manifest tmp/wuxi_dataset_manifest/dataset_manifest.json \
  --output_dir tmp/wuxi_stereo_quality
```

Ropedia：

```bash
python scripts/quality/check_stereo_video_quality.py \
  --manifest tmp/ropedia_dataset_manifest/dataset_manifest.json \
  --output_dir tmp/ropedia_stereo_quality
```

输出：

```text
tmp/<dataset>_stereo_quality/
  stereo_quality_report.json
  stereo_quality_samples.csv
  contact_sheet.png
```

核心检查：

```text
brightness mean/std
Laplacian variance 清晰度
过曝/欠曝比例
左右亮度差
ORB matches
ORB vertical error
```

进入下一步标准：

```text
stereo_quality_report.json
  summary.passed == true
```

### 局部异常段质检

用途：找出局部坏帧/坏片段，避免平均值掩盖几秒钟的曝光、模糊、左右不同步或极线异常。

```bash
python scripts/quality/check_stereo_video_quality_segments.py \
  --manifest tmp/wuxi_dataset_manifest/dataset_manifest.json \
  --output_dir tmp/wuxi_stereo_quality_segments \
  --sample_every_s 1.0 \
  --orb_every_s 5.0
```

Ropedia：

```bash
python scripts/quality/check_stereo_video_quality_segments.py \
  --manifest tmp/ropedia_dataset_manifest/dataset_manifest.json \
  --output_dir tmp/ropedia_stereo_quality_segments \
  --sample_every_s 1.0 \
  --orb_every_s 5.0
```

输出：

```text
tmp/<dataset>_stereo_quality_segments/
  bad_segments.json
  bad_frames.csv
  all_sampled_frames.csv
  contact_sheet_bad.png
```

进入下一步标准：

```text
bad_segments.json
  passed == true
```

如果不通过，先查看 `contact_sheet_bad.png` 和 `bad_frames.csv`，判断是曝光、模糊、纹理不足、左右亮度差，还是 ORB vertical error 异常。

### Ropedia

默认不重跑 HaMeR，只从已有候选、已有 depth 继续生成 alignment、tracking、RRD。

```bash
bash scripts/run_ropedia_pipeline.sh
```

如需重跑 HaMeR：

```bash
RUN_HAMER=1 bash scripts/run_ropedia_pipeline.sh
```

常用覆盖参数：

```bash
START=0 \
NUM_FRAMES=300 \
RUN_HAMER=0 \
RUN_RECOVER=1 \
RUN_ALIGN=1 \
RUN_TRACK=1 \
RUN_RRD=1 \
bash scripts/run_ropedia_pipeline.sh
```

### Wuxi

标准 300 帧全流程：

```bash
RUN_FOUNDATION=1 \
FOUNDATION_OVERWRITE=1 \
START=0 \
NUM_FRAMES=300 \
bash scripts/run_wuxi_pipeline.sh
```

如果 FoundationStereo depth 已经存在，只重跑后续手部流程：

```bash
RUN_FOUNDATION=0 \
RUN_HAMER=1 \
RUN_RECOVER=1 \
RUN_ALIGN=1 \
RUN_TRACK=1 \
RUN_RRD=1 \
START=0 \
NUM_FRAMES=300 \
bash scripts/run_wuxi_pipeline.sh
```

当前确认过的新标准 Wuxi 输出：

```text
tmp/wuxi_hand_vis_0_299_wuxiK_finger_orient/wuxi_hand_vis.rrd
```

打开：

```bash
rerun /path/to/roboego-hand-vis/tmp/wuxi_hand_vis_0_299_wuxiK_finger_orient/wuxi_hand_vis.rrd
```

## 3. 公共参数

公共参数集中在：

```text
scripts/pipeline/common.sh
```

当前标准值：

```text
START=0
NUM_FRAMES=300
SCENE_POINT_STRIDE=4
MAX_SCENE_POINTS=90000

RECOVERY_MIN_AREA_RATIO=0.0007
RECOVERY_MIN_CENTER_Y_RATIO=0.35
RECOVERY_MAX_MEDIAN_DEPTH_M=1.2

MAX_INTERP_GAP=10
SMOOTH_ALPHA=0.35
JUMP_THRESHOLD_M=0.08
MESH_ERROR_THRESHOLD_M=0.05
MIN_VALID_DEPTH_RATIO=0.5

MESH_MASK_SOURCE=hamer_overlay_diff
MESH_MASK_PADDING_PX=2
OVERLAY_DIFF_THRESHOLD=8
OVERLAY_DIFF_SEARCH_MARGIN_PX=24
MASK_DEPTH_BAND_HALF_WIDTH_M=0.12
DEPTH_LIFT_WINDOW_RADIUS_PX=2
DEPTH_LIFT_MAX_DELTA_M=0.12

SHAPE_PRESERVED_DEPTH_SOURCE=finger
ORIENTATION_AWARE_COMPENSATION=1
ORIENT_DEPTH_BIAS_M=0.035
ORIENT_SURFACE_MIN_M=0.005
ORIENT_SURFACE_MAX_M=0.020
ORIENT_RAY_BLEND=0.3
```

关键含义：

```text
RECOVERY_MIN_AREA_RATIO:
  放宽小手候选恢复阈值。当前 0.0007 用于保留远处或较小的 ego hand 候选。

MAX_INTERP_GAP:
  smooth / interpolated 最多补 10 帧以内的缺失段。更长缺失不强行补。

MESH_MASK_SOURCE=hamer_overlay_diff:
  使用 RGB 原图和 HaMeR 官方 overlay 的差异区域，得到实际渲染出来的手部 2D mask。

SHAPE_PRESERVED_DEPTH_SOURCE=finger:
  final MANO placement 的 depth anchor 优先使用 finger / palm-side 区域，而不是整手 mask median。

ORIENTATION_AWARE_COMPENSATION=1:
  根据 MANO 手掌朝向估计 depth surface 到 skeleton / MANO 中心的补偿。
```

## 4. Step 2: FoundationStereo Depth

脚本：

```text
scripts/run_foundation_stereo_video.py
```

标准入口现在读取 `dataset_manifest.json`，不再手写左右视频和内参路径。Wuxi 通过 `scripts/run_wuxi_pipeline.sh` 自动调用。Ropedia 当前一般复用已有 depth，也可以用同一入口重跑。

### 输入

```text
--manifest:
  标准 dataset_manifest.json。脚本从这里读取 left/right video、left K、baseline。

--foundation_root:
  FoundationStereo 仓库路径。

--ckpt:
  FoundationStereo 权重。
```

Wuxi：

```bash
python scripts/run_foundation_stereo_video.py \
  --manifest tmp/wuxi_dataset_manifest/dataset_manifest.json \
  --output_dir tmp/wuxi_foundation_depth_continuous_000000_000299 \
  --start 0 \
  --num_frames 300 \
  --scale 0.5 \
  --valid_iters 32 \
  --overwrite
```

Ropedia：

```bash
python scripts/run_foundation_stereo_video.py \
  --manifest tmp/ropedia_dataset_manifest/dataset_manifest.json \
  --output_dir tmp/ropedia_foundation_depth_continuous_000000_000299 \
  --start 0 \
  --num_frames 300 \
  --scale 1.0 \
  --valid_iters 32 \
  --overwrite
```

### 关键参数

```text
--start:
  起始帧。

--num_frames:
  处理帧数。

--scale:
  FoundationStereo 输入缩放。当前 Wuxi 标准为 0.5。

--valid_iters:
  FoundationStereo 迭代次数。当前为 32。

--max_depth_m:
  最大有效深度，当前为 4.0m。

--overwrite:
  覆盖已存在 depth 输出。
```

### 输出

```text
tmp/<dataset>_foundation_depth_continuous_000000_000299/
  frame_000000_foundation_depth_m.npy
  frame_000001_foundation_depth_m.npy
  ...
  vis/frame_000000_depth_vis.jpg
  foundation_stereo_video_meta.json
```

其中 `*_depth_m.npy` 是米制深度，坐标参考左相机。

## 5. Step 3: Depth 质量检查

脚本：

```text
scripts/depth/check_depth_quality.py
```

用途：

```text
确认 depth 是否和 left RGB 对齐
确认 depth 尺度是否合理
生成 RGB-depth overlay
生成 point cloud preview
计算 warp error
给低质量帧打 warning
```

Wuxi：

```bash
python scripts/depth/check_depth_quality.py \
  --manifest tmp/wuxi_dataset_manifest/dataset_manifest.json \
  --depth_dir tmp/wuxi_foundation_depth_continuous_000000_000299 \
  --output_dir tmp/wuxi_depth_quality_000000_000299 \
  --stereo_quality_report tmp/wuxi_stereo_quality_segments/bad_segments.json \
  --start 0 \
  --num_frames 300
```

Ropedia：

```bash
python scripts/depth/check_depth_quality.py \
  --manifest tmp/ropedia_dataset_manifest/dataset_manifest.json \
  --depth_dir tmp/ropedia_foundation_depth_continuous_000000_000299 \
  --output_dir tmp/ropedia_depth_quality_000000_000299 \
  --stereo_quality_report tmp/ropedia_stereo_quality_segments/bad_segments.json \
  --start 0 \
  --num_frames 300
```

输出：

```text
tmp/<dataset>_depth_quality_000000_000299/
  depth_quality_report.json
  depth_quality_frames.csv
  vis/
    frame_000000_depth_vis.jpg
    frame_000000_depth_overlay.jpg
  pointcloud_preview/
    frame_000000_pointcloud_preview.ply
```

核心字段：

```text
valid_depth_ratio:
  当前帧有效深度比例。

depth_min_m / depth_median_m / depth_max_m:
  当前帧深度范围，单位米，用于判断尺度是否离谱。

warp_l1_mean:
  用 depth 把右图 warp 到左图后的颜色误差。越低越好；遮挡区域会自然偏高。

edge_warp_l1_mean:
  边缘区域 warp error。手、物体边界、遮挡和低纹理区域通常更高。

hand_mask_depth_median:
  预留字段。没有手部候选时为 null；接入 HaMeR 后可统计手部 mask 内 depth。

warnings:
  low_valid_depth_ratio / high_warp_error / high_edge_warp_error / missing_depth。
  如果传入 stereo_quality_report，还会追加 stereo_quality_<reason>，例如
  stereo_quality_sharpness_low、stereo_quality_p90_abs_dy。
```

注意：

```text
depth 质量检查不直接修改 depth。
它只判断哪些帧或哪些现象需要后续 hand-depth 对齐时降低信任。
```

## 6. Step 4: HaMeR 候选和 Ego Hand 初筛

入口：

```text
scripts/pipeline/01_run_hamer_candidates.sh
```

实际调用：

```text
scripts/hand/run_hamer_ego_filter.py
```

### 输入

```text
DATASET_MANIFEST:
  标准 dataset_manifest.json。脚本从这里读取 left video、dataset_name、sequence_name、fps、resolution。

DEPTH_DIR:
  FoundationStereo depth 输出目录。

DEPTH_QUALITY_REPORT:
  可选。depth 质量检查报告。若存在，会把每帧 depth/stereo warning 写进 ego_hand_candidates.json。

HAMER_ROOT:
  HaMeR 仓库目录。
```

手动运行示例：

```bash
python scripts/hand/run_hamer_ego_filter.py \
  --manifest tmp/wuxi_dataset_manifest/dataset_manifest.json \
  --depth_dir tmp/wuxi_foundation_depth_continuous_000000_000299 \
  --depth_quality_report tmp/wuxi_depth_quality_000000_000299/depth_quality_report.json \
  --output_dir tmp/wuxi_hamer_ego_filter_000000_000299_tuned \
  --hamer_root /path/to/hamer \
  --start 0 \
  --num_frames 300 \
  --body_detector regnety \
  --batch_size 1 \
  --save_mesh
```

### 关键参数

```text
HAMER_BODY_DETECTOR=regnety
HAMER_BATCH_SIZE=1

RECOVERY_MIN_AREA_RATIO=0.0007
RECOVERY_MIN_CENTER_Y_RATIO=0.35
RECOVERY_MAX_MEDIAN_DEPTH_M=1.2
```

HaMeR per-frame 运行，不默认使用相邻帧信息。所以相邻帧偶尔丢手、重复检测、左右手 ID 跳变是正常现象，后面用 recover / tracking / interpolation 处理。

### 输出

```text
tmp/<dataset>_hamer_ego_filter_..._tuned/
  ego_hand_candidates.json
  meshes/
    frame_000000_candidate_00.obj
    ...
  hamer_overlay/
    frame_000000_hamer_overlay.jpg
    ...
  contact_sheet...
```

主要字段：

```text
ego_hand_candidates.json:
  manifest_path / dataset_name / sequence_name / reference_camera
  video / video_info
  depth_dir / depth_quality_report
  hamer_config / filter_config
  每帧所有 HaMeR 候选
  ego_left / ego_right 初筛结果
  depth_quality_warnings
  bbox、面积、中心位置、depth 统计、filter/reject reason
  mesh obj 路径
  HaMeR overlay 路径
```

## 7. Step 5: Recover 被误过滤的候选

入口：

```text
scripts/pipeline/02_recover_candidates.sh
```

实际调用：

```text
scripts/recover_ego_hand_candidates.py
```

### 输入

```text
HAMER_CANDIDATES_JSON:
  Step 2 输出的 ego_hand_candidates.json。
```

### 关键参数

```text
RECOVERY_MIN_AREA_RATIO=0.0007
RECOVERY_MIN_CENTER_Y_RATIO=0.35
RECOVERY_MAX_MEDIAN_DEPTH_M=1.2
```

作用：

```text
如果某帧 ego_left / ego_right 缺失，
但 HaMeR 其实检测到了小候选，
且候选位置、面积、深度还合理，
则恢复为 recovered。
```

### 输出

```text
tmp/<dataset>_hamer_ego_filter_..._recovered_area0007_temporal_gated/
  ego_hand_candidates_recovered.json
  contact_sheet...
```

状态含义：

```text
normal:
  原始筛选就通过的手。

recovered:
  HaMeR 检测到了，但第一轮 filter 过滤掉，第二轮按放宽规则恢复。

missing:
  当前帧没有可用候选，后面才可能由 smooth / interpolated 补。
```

## 8. Step 6: Mesh-Depth Alignment Report

入口：

```text
scripts/hand/align_mesh_depth.py
```

内部实现：

```text
scripts/hand/mesh_depth_alignment_impl.py
```

这是当前流程最核心的一步。标准入口负责从 manifest 读取 left video / intrinsics，并把 candidates 按 start/num_frames 过滤后交给内部实现。

核心逻辑已经逐步拆到模块：

```text
roboego_hand_vis/egohand/depth_anchor.py:
  统一选择 core/mask/bbox/finger depth anchor。

roboego_hand_vis/egohand/mesh_placement.py:
  shape-preserved MANO mesh placement，保留 HaMeR 手形状，只做全局放置。

roboego_hand_vis/egohand/orientation.py:
  MANO joint regressor、21 joints、orientation-aware compensation 调用封装。

roboego_hand_vis/egohand/rerun_logging.py:
  标准 Rerun entity 路径和部分 mesh/2D points logging helper。
```

手动运行示例：

```bash
python scripts/hand/align_mesh_depth.py \
  --manifest tmp/wuxi_dataset_manifest/dataset_manifest.json \
  --candidates_json tmp/wuxi_hamer_ego_filter_000000_000299_tuned/ego_hand_candidates.json \
  --depth_dir tmp/wuxi_foundation_depth_continuous_000000_000299 \
  --output_dir tmp/wuxi_hand_depth_alignment_000000_000599 \
  --start 0 \
  --num_frames 600 \
  --no_save_rrd
```

### 输入

```text
CANDIDATES_JSON:
  HaMeR ego filter 输出的 ego_hand_candidates.json。

DEPTH_DIR:
  FoundationStereo depth 输出目录。

MANIFEST:
  标准 dataset_manifest.json。脚本从这里读取 left video 和 intrinsics。

HAMER_OVERLAY_DIR:
  可选。默认使用 candidates_json 同级目录下的 hamer_overlay。

MANO_MODEL_DIR:
  MANO_LEFT.pkl / MANO_RIGHT.pkl 所在目录。
```

### 标准方法

```text
RGB 原图
HaMeR overlay 图
        |
        | absdiff
        v
HaMeR 实际渲染手部区域 mask
        |
        v
mask 内从 depth 取 hand depth points
        |
        v
上半部 / finger-side mask 取 depth anchor
        |
        v
保持 HaMeR / MANO 原始手形状，只做 global placement
        |
        v
orientation-aware surface-to-skeleton compensation
        |
        v
current_best_mesh
```

### 关键参数

```text
--mesh_mask_source hamer_overlay_diff:
  用 RGB 与 HaMeR overlay 的差异得到真实渲染手部 mask。

--shape_preserved_depth_source finger:
  final mesh placement 使用 finger/palm-side 深度点，而不是整手 median。

--orientation_aware_compensation:
  开启朝向感知补偿。

--orientation_depth_bias_m 0.01:
  固定补偿基础量，单位米。

--orientation_surface_min_m 0.005:
  手侧面时的最小 surface-to-skeleton 补偿。

--orientation_surface_max_m 0.010:
  手掌/手背正对相机时的最大 surface-to-skeleton 补偿。

--orientation_ray_blend 0.3:
  30% 沿相机射线补偿，70% 类似 z 方向补偿。

--mask_depth_band_half_width_m 0.12:
  hand mask 内深度带宽过滤，减少物体、袖子、小臂干扰。

--depth_lift_window_radius_px 2:
  从 mesh 投影点附近取 depth 时的像素窗口。

--depth_lift_max_delta_m 0.12:
  过滤离 mask median depth 太远的异常 lifted vertex。
```

### 输出

```text
tmp/<dataset>_hand_depth_alignment_.../
  hand_depth_mesh_alignment_report.json
  hand_depth_mesh_alignment_summary.json
  inputs/ego_hand_candidates_<start>_<end>.json
  ropedia_hamer_depth_alignment.rrd  # 未使用 --no_save_rrd 时生成
```

`hand_depth_mesh_alignment_report.json` 重要字段：

```text
intrinsics:
  本次用于投影、反投影、点云生成的相机内参。

shape_preserved_depth_source:
  当前标准应为 finger。

orientation_aware_compensation:
  当前标准应 enabled=true，并记录具体参数。

frames[].hands[]:
  每帧每只手的状态、bbox、depth 统计、mesh-depth 误差、transform。

shape_preserved_fit_transform:
  shape-preserved MANO placement 的 global translation 信息。

target_depth_points_source:
  应优先为 finger_upper_mask_points。

orientation_aware_compensation:
  每只手实际补偿方向和补偿量。
```

注意：orientation-aware compensation 会把 MANO skeleton/mesh center 从可见 depth surface 往内部补偿。因此 `mesh2depth` 指标不一定越小越好；过于贴近可见表面反而可能让 3D joints 更靠近相机。

## 9. Step 7: Tracking / Fill Missing / Smooth

入口：

```text
scripts/pipeline/04_track_fill_missing.sh
```

实际调用：

```text
scripts/hand/track_fill_smooth.py
```

### 输入

```text
ALIGNMENT_REPORT:
  Step 6 输出的 hand_depth_mesh_alignment_report.json。

DEPTH_DIR:
  FoundationStereo depth 输出目录。

DATASET_MANIFEST:
  标准 dataset_manifest.json。脚本从这里读取 left video、dataset_name、sequence_name、fps、resolution。
```

手动运行示例：

```bash
python scripts/hand/track_fill_smooth.py \
  --manifest tmp/wuxi_dataset_manifest/dataset_manifest.json \
  --alignment_report tmp/wuxi_hand_depth_alignment_000000_000299/hand_depth_mesh_alignment_report.json \
  --depth_dir tmp/wuxi_foundation_depth_continuous_000000_000299 \
  --output_dir tmp/wuxi_ego_hand_tracking_000000_000299 \
  --start 0 \
  --num_frames 600 \
  --max_interp_gap 10 \
  --smooth_alpha 0.35 \
  --apply_recovered_depth_overrides \
  --rrd_start 0 \
  --rrd_num_frames 30
```

### 当前策略

```text
normal:
  JSON status 为 detected。信任当前帧 alignment 结果。

recovered:
  JSON status 为 recovered。候选来自恢复规则。
  深度使用相邻 normal/recovered 稳健参考，避免局部 depth anchor 乱飞。

smooth / interpolated:
  JSON status 为 interpolated，统一 layer 名为 smooth。
  只补缺失帧，不覆盖正常 normal/recovered。
  缺失间隔 <= MAX_INTERP_GAP 才补。

current:
  每帧每只手最终给下游使用的统一结果。normal/recovered 原样保留，缺失帧才使用 smooth/interpolated。

final:
  基于 current 轨迹做 weighted temporal smoothing。
  normal 权重大、修正幅度小；recovered/smooth 权重更低、允许更大修正。
  final 只作为最终平滑层输出，不覆盖 current/normal/recovered/smooth。
```

### 关键参数

```text
--max_interp_gap 10:
  最多补 10 帧缺失。

--smooth_alpha 0.35:
  translation 平滑强度。

--jump_threshold_m 0.08:
  跳变检测阈值。

--mesh_error_threshold_m 0.05:
  mesh-depth 误差异常阈值。

--min_valid_depth_ratio 0.5:
  有效 depth 比例过低时标记低质量。

--apply_recovered_depth_overrides:
  recovered 手的 z 使用相邻可靠帧插值/继承。

--no-drop_unstable_recovered:
  当前标准不强制丢弃 recovered，主要在 Rerun 中分层查看。

--final_smoothing_window 5:
  final temporal smoothing 窗口。设为 0 可关闭 final 层。

--rrd_start / --rrd_num_frames:
  只控制 RRD 可视化帧范围，不影响 JSON/CSV 的处理帧数。
  例如 --num_frames 600 --rrd_start 0 --rrd_num_frames 30 表示处理 600 帧，但只保存前 30 帧到 Rerun。

--no_save_rrd:
  只输出 tracking JSON/CSV/debug frames，不生成 RRD。
```

### 输出

```text
tmp/<dataset>_ego_hand_tracking_.../
  ego_hand_tracks.json
  ego_hand_tracks.csv
  contact_sheet_tracking_debug.jpg
  debug_frames/
  ropedia_ego_hands_tracked_smoothed.rrd
```

RRD 主要路径：

```text
camera_opencv/camera/left/rgb
camera_opencv/camera/left/depth_vis
camera_opencv/depth/full_scene_points

camera_opencv/hands/ego_left/current
camera_opencv/hands/ego_left/normal
camera_opencv/hands/ego_left/recovered
camera_opencv/hands/ego_left/smooth
camera_opencv/hands/ego_left/final

camera_opencv/hands/ego_right/current
camera_opencv/hands/ego_right/normal
camera_opencv/hands/ego_right/recovered
camera_opencv/hands/ego_right/smooth
camera_opencv/hands/ego_right/final
```

`current` 是每帧最终使用的手；`normal/recovered/smooth` 是来源分层，方便排查 recovered 或 smooth 是否异常；`final` 是 weighted temporal smoothing 后的最终候选。

`ego_hand_tracks.json` 顶层字段：

```text
manifest_path / dataset_name / sequence_name / reference_camera
alignment_report / depth_dir / video
status_layers:
  detected -> normal
  recovered -> recovered
  interpolated -> smooth
  current -> 下游统一读取的每帧最终手结果
parameters
summary
tracks
```

主要统计：

```text
detected:
  normal，当前帧直接检测/对齐得到的帧数。

recovered:
  通过恢复规则找回的帧数。

interpolated:
  smooth，缺失后由相邻帧补出来的帧数。

missing:
  超出 max gap 或无可靠邻帧，未补齐。

anomaly:
  跳变或 mesh-depth 质量异常。

present_ratio:
  最终可用比例。
```

## 10. Step 8: Final RRD 可视化

入口：

```text
scripts/pipeline/05_export_final_rerun.sh
```

实际调用：

```text
scripts/hand/export_final_rerun.py
```

### 输入

```text
TRACKS_JSON:
  Step 5 输出的 ego_hand_tracks.json。

ALIGNMENT_REPORT:
  Step 4 输出的 hand_depth_mesh_alignment_report.json，用于读取相机内参和 orientation compensation 参数。

DEPTH_DIR:
  FoundationStereo depth。

RRD_START / RRD_NUM_FRAMES:
  只控制 Rerun 导出的帧范围，不影响前面已经处理的完整轨迹。
```

### 关键参数

```text
--final_smoothing_window:
  最终轨迹平滑窗口。normal/recovered 为主，smooth 主要补缺失和稳定轨迹。

--scene_point_stride / --max_scene_points:
  控制全场景点云密度，数值越小点越密，RRD 越大越卡。

--output_rrd:
  最终 Rerun 文件路径。
```

### 输出

```text
tmp/<dataset>_hand_vis_.../
  <dataset>_hand_vis.rrd
```

Rerun 主要字段：

```text
camera_opencv/camera/left/rgb:
  左目 RGB 原图。

camera_opencv/camera/left/depth_vis:
  depth 可视化图，用于快速看深度层次。

camera_opencv/depth/full_scene_points:
  全画面 depth 点云，用于判断手是否在合理场景深度。

camera_opencv/hands/ego_left/final:
  当前最终左手 MANO mesh。

camera_opencv/hands/ego_right/final:
  当前最终右手 MANO mesh。

camera_opencv/camera/left/rgb/overlay/final/ego_left/mesh_projected_points:
  左手 final mesh 投影回 RGB 的 2D 点。

camera_opencv/camera/left/rgb/overlay/final/ego_right/mesh_projected_points:
  右手 final mesh 投影回 RGB 的 2D 点。
```

## 11. Ropedia 官方精度评估

脚本：

```text
scripts/export_and_evaluate_ropedia_hand_joints.py
```

作用：

```text
读取 Ropedia annotation.hdf5 官方 hand_mocap。
读取我们流程输出的 current mesh。
统一到 camera_opencv。
由 MANO joint regressor 得到 21 joints。
计算 MPJPE、wrist、palm MCP、fingertip、non-wrist、jitter 等指标。
```

当前建议指标：

```text
MPJPE:
  21 joints 平均误差。

wrist error:
  wrist 单点误差。

palm MCP error:
  [5, 9, 13, 17] 掌指关节误差。

fingertip error:
  [4, 8, 12, 16, 20] 指尖误差。

non-wrist MPJPE:
  joints[1:21]，排除 wrist 后的平均误差。

wrist-to-palm length difference:
  判断 wrist / palm 相对几何是否偏长或偏短。

temporal jitter:
  相邻帧 3D joints / hand center 跳变量。

missing rate:
  未检测、未恢复、无法插值的比例。
```

Wuxi 当前没有官方 annotation，因此主要依赖：

```text
Rerun 肉眼检查
RGB overlay 是否贴手
hand_pointcloud_final 是否覆盖真实手
current_best_mesh 是否在全局点云合理深度
jump / anomaly / missing 统计
```

## 12. Ropedia 与 Wuxi 差异

### Ropedia

```text
有官方 hand_mocap annotation。
可以做 MPJPE / joints / temporal jitter 定量对比。
stereo_left/right 已基本适合 FoundationStereo。
默认从 annotation.hdf5 的 calibration/cam01 读取 stereo K 和 baseline。
```

### Wuxi

```text
没有官方 hand annotation。
必须使用 data/wuxi/camera_intrinsics.json。
必须由 stereo_left.mp4 / stereo_right.mp4 跑 FoundationStereo。
GoPro/广角/ISP/场景纹理会影响 depth 稳定性。
finger anchor 和 orientation-aware compensation 已作为标准流程开启。
```

## 13. 标准 Wuxi 300 帧复跑命令

如果需要复现当前确认版本：

```bash
cd /path/to/roboego-hand-vis
source /path/to/conda/etc/profile.d/conda.sh
conda activate ego

RUN_FOUNDATION=0 \
RUN_HAMER=0 \
RUN_RECOVER=0 \
RUN_ALIGN=1 \
RUN_TRACK=1 \
RUN_RRD=1 \
START=0 \
NUM_FRAMES=300 \
VIDEO=/path/to/roboego-hand-vis/data/wuxi/stereo_left.mp4 \
RIGHT_VIDEO=/path/to/roboego-hand-vis/data/wuxi/stereo_right.mp4 \
INTRINSICS_JSON=/path/to/roboego-hand-vis/data/wuxi/camera_intrinsics.json \
INTRINSICS_CAMERA=left \
INTRINSICS_RESIZE_SCALE=1.0 \
DEPTH_DIR=tmp/wuxi_foundation_depth_continuous_000000_000299 \
HAMER_OUT=tmp/wuxi_hamer_ego_filter_0_299_tuned \
HAMER_CANDIDATES_JSON=tmp/wuxi_hamer_ego_filter_0_299_tuned/ego_hand_candidates.json \
RECOVER_OUT=tmp/wuxi_hamer_ego_filter_0_299_recovered_area0007_temporal_gated \
CANDIDATES_JSON=tmp/wuxi_hamer_ego_filter_0_299_recovered_area0007_temporal_gated/ego_hand_candidates_recovered.json \
HAMER_OVERLAY_DIR=tmp/wuxi_hamer_ego_filter_0_299_tuned/hamer_overlay \
SHAPE_PRESERVED_DEPTH_SOURCE=finger \
ORIENTATION_AWARE_COMPENSATION=1 \
ORIENT_DEPTH_BIAS_M=0.035 \
ORIENT_SURFACE_MIN_M=0.005 \
ORIENT_SURFACE_MAX_M=0.020 \
ORIENT_RAY_BLEND=0.3 \
ALIGN_OUT=tmp/wuxi_hand_depth_alignment_0_299_wuxiK_finger_orient \
ALIGNMENT_REPORT=tmp/wuxi_hand_depth_alignment_0_299_wuxiK_finger_orient/hand_depth_mesh_alignment_report.json \
TRACK_OUT=tmp/wuxi_ego_hand_tracking_0_299_wuxiK_finger_orient \
TRACKS_JSON=tmp/wuxi_ego_hand_tracking_0_299_wuxiK_finger_orient/ego_hand_tracks.json \
RRD_OUT=tmp/wuxi_hand_vis_0_299_wuxiK_finger_orient \
SAVE_RRD=tmp/wuxi_hand_vis_0_299_wuxiK_finger_orient/wuxi_hand_vis.rrd \
bash scripts/run_wuxi_pipeline.sh
```

输出：

```text
tmp/wuxi_hand_depth_alignment_0_299_wuxiK_finger_orient/hand_depth_mesh_alignment_report.json
tmp/wuxi_ego_hand_tracking_0_299_wuxiK_finger_orient/ego_hand_tracks.json
tmp/wuxi_hand_vis_0_299_wuxiK_finger_orient/wuxi_hand_vis.rrd
```

## 14. 当前 Wuxi 300 帧结果摘要

本次确认版本：

```text
RRD:
  tmp/wuxi_hand_vis_0_299_wuxiK_finger_orient/wuxi_hand_vis.rrd

report:
  tmp/wuxi_hand_vis_0_299_wuxiK_finger_orient/hand_depth_mesh_alignment_report.json

tracking:
  tmp/wuxi_ego_hand_tracking_0_299_wuxiK_finger_orient/ego_hand_tracks.json
```

确认字段：

```text
num_frames = 300
intrinsics.fx = 1032.794934191615
intrinsics.fy = 1044.7178468341592
intrinsics.cx = 811.5
intrinsics.cy = 615.5
shape_preserved_depth_source = finger
orientation_aware_compensation.enabled = true
```

Tracking 摘要：

```text
ego_left:
  detected: 300
  missing: 0
  present_ratio: 1.0

ego_right:
  detected: 291
  interpolated: 9
  missing: 0
  anomaly: 2
  present_ratio: 1.0
```

观察注意：

```text
Wuxi 后半段局部 depth / finger anchor 有波动。
orientation-aware compensation 是 skeleton/center 补偿，不是让 mesh 完全贴 depth surface。
因此要同时看 RGB overlay、hand_pointcloud_final、full_scene_points 和 current_best_mesh。
```

## 15. 推荐检查顺序

每次换数据集或调参数，按下面顺序看：

```text
1. RGB 原图是否正常。
2. HaMeR overlay 是否贴 RGB 手部。
3. depth_vis 是否有明显异常。
4. full_scene_points 是否形成合理场景。
5. hand_pointcloud_final 是否覆盖真实手。
6. depth_points_finger_anchor 是否落在手掌/手指侧。
7. current_best_mesh 是否保持 MANO 手形状，并位于合理 3D 深度。
8. normal / recovered / smooth 是否分层合理。
9. tracking_summary 中 missing、interpolated、anomaly 是否可接受。
10. Ropedia 再看 MPJPE / grouped joint errors。
```
