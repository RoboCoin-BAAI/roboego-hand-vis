# RoboEgo Hand Vis

[English](README.md) | 简体中文

RoboEgo Hand Vis 是一个基于 Kedro 的处理流水线，用于处理头戴式双目第一人称手部数据。当前架构按阶段组织流水线：

```text
quality -> semantic -> datasets -> restoration -> depth -> egohand -> objects -> privacy
```

主要用途是将原始头带采集会话处理为筛选后的手物交互时间窗口，输出双目视频、校正后的数据集清单、深度、HaMeR 手部网格、过滤后的 21 关节手部轨迹，以及 SAM3 物体掩码。

L1/L2/L3 数据登记和共享目录下载状态使用系统 PostgreSQL。固定的建用户、建库、配置连接和建表步骤见[数据库初始化说明](roboego_hand_vis/database/README.md#初始化步骤系统-postgresql)。

L1 的直接 KS3 下载和数据库任务下载见[数据下载说明](roboego_hand_vis/download_data/README.md)。

头环基础质检的原有命令和独立数据库批处理入口见[基础质检说明](roboego_hand_vis/raw_headband_integrity/README.md)。

左右目视频粗质检的原有命令和独立数据库批处理入口见[视频粗质检说明](roboego_hand_vis/video_simple_quality/README.md)。

可选的手部采样检测及独立数据库批处理入口见[手部检测说明](roboego_hand_vis/detect_hand/README.md)。

去畸变、立体矫正和独立数据库批处理入口见[立体矫正说明](roboego_hand_vis/stereo_rectification/README.md)。

千问逐帧标注和独立数据库批处理入口见[千问标注说明](roboego_hand_vis/qwen_annotation/README.md)。

标注后字段整理和独立数据库批处理入口见[字段整理说明](roboego_hand_vis/post_annotation_actions/README.md)。

深度前质量门禁和数据库批处理入口见[深度门禁说明](roboego_hand_vis/stereo_depth_gate/README.md)。

Fast-FoundationStereo 深度估计和数据库批处理入口见[深度估计说明](roboego_hand_vis/fast_foundation_depth/README.md)。

MINT 与 EgoHandKit 手部估计和数据库批处理入口见[手部估计说明](roboego_hand_vis/hand/README.md#l1-数据库入口)。

深度缓存归档和数据库批处理入口见[深度归档说明](roboego_hand_vis/archive_depth_cache/README.md)。

视频人脸打码和数据库批处理入口见[视频打码说明](roboego_hand_vis/video_privacy_blur/README.md)。

OpenVINS 头部轨迹估计和数据库批处理入口见[VIO 说明](roboego_hand_vis/vio/README.md)。

各数据库入口支持 `--table l1data|l2data|l3data`，默认 `l1data`。目标表需要具备相应阶段字段，见[选表说明](roboego_hand_vis/database/README.md#数据库入口选表)。

需要在本地盘处理并按阶段回传共享盘时，使用 `--local-root`（基础质检自动复制缺失数据），见[本地工作目录说明](roboego_hand_vis/database/WORKSPACE.md)。

## 环境配置

项目通常在 `ego` 环境中运行：

```bash
cd /path/to/roboego-hand-vis
source /path/to/conda/etc/profile.d/conda.sh
conda activate ego

pip install -e ".[dev]"
```

预期的本地 Python 环境：

```text
ego      主 Kedro 流水线、质量检查、语义衔接、数据集处理、手部过滤
py312    Fast-FoundationStereo 和 SAM3 封装
hamer    HaMeR 推理
realvdeblur  可选的 RealVDeblur 视频修复
```

完整的机器与环境对应关系见[流水线环境配置](docs/deployment/environments.md)。部分节点依赖以下外部仓库或模型文件：

```text
/path/to/RoboEgoPipe
/path/to/Fast-FoundationStereo
/path/to/hamer
/path/to/do-as-i-do/sam3/sam3.pt
/path/to/RealVDeblur
```

使用 Qwen 标注前，请设置：

```bash
export DASHSCOPE_API_KEY=...
```

## OpenVINS 视觉惯性里程计（VIO）

VIO 目前是独立的离线适配器，尚未作为节点集成到 Kedro 手部流水线中。它将带时间戳的原始双目图像与 IMU 测量值回放到 OpenVINS，并记录 IMU 位姿和左相机位姿。它不依赖 Qwen、其他视觉语言模型（VLM）的凭据、深度、HaMeR 或 SAM3。

### 输入

VIO 会话必须包含以下原始会话目录结构：

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

请使用未经校正的原始相机视频。运行器直接加载标定后的等距投影相机模型。`left/right` 对应 OpenVINS 的 `cam0/cam1`；完整会话中可以保留 `bleft/bright` 和音频，但当前双相机 VIO 适配器不会读取这些数据。

估计前，适配器会校验视频尺寸与标定是否一致，按时间戳匹配双目帧，去除重复时间戳，检查 IMU 覆盖范围和数据间隔，将 Kalibr 的 `T_cam_imu` 转换为 OpenVINS 的 `T_imu_cam`，并写入 OpenVINS 使用的 OpenCV-YAML 配置。

### 依赖

估计器进程需要：

```text
Ubuntu 22.04 / ROS 2 Humble       /opt/ros/humble/setup.bash
OpenVINS ROS 2 build              <OPENVINS_ROOT>/install_ros2/setup.bash
OpenVINS packages                 ov_core, ov_init, ov_msckf
原生构建依赖         Eigen3, Boost, Ceres
系统 Python                       rclpy, sensor_msgs, geometry_msgs,
                                  numpy, OpenCV, PyYAML
```

运行器使用 `/usr/bin/python3` 以兼容 ROS Python。Rerun 导出和可视化则使用项目环境。标准的 OpenVINS 构建命令如下：

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

英文文档所述工作站已在 `/home/user/open_vins/install_ros2` 安装 OpenVINS。其用户级 Ceres 库位于 `/home/user/.local/openvins-deps/usr`，运行器会自动添加该库目录。可通过以下命令验证环境：

```bash
source /opt/ros/humble/setup.bash
source /home/user/open_vins/install_ros2/setup.bash
/usr/bin/python3 -c 'import cv2, numpy, yaml, rclpy; import geometry_msgs, sensor_msgs'
ros2 pkg executables ov_msckf
```

可通过 `OPENVINS_ROOT`、`OPENVINS_INSTALL`、`OPENVINS_DEPS`、`SYSTEM_PYTHON` 和 `ROS_DOMAIN_ID` 修改运行器默认配置。`ffmpeg` 为可选依赖，仅生成相机画面与轨迹对比 MP4 时需要。

### 运行一分钟测试片段

`06:49-07:49` 的完整测试片段位于：

```text
/home/user/ego_data/session_20260804_224442_clip_0649_60s/session
```

运行并写入新的输出目录：

```bash
cd /home/user/roboego-hand-vis

scripts/vio/run_openvins_session.sh \
  --session /home/user/ego_data/session_20260804_224442_clip_0649_60s/session \
  --output tmp/openvins_clip_0649_60s \
  --rate 1.0
```

常用选项：

```text
--start_s SECONDS       相对于首对同步双目帧的时间偏移
--duration_s SECONDS    仅处理指定时长；省略则处理整个会话
--rate RATE             每秒实际时间回放的数据集秒数
--no-downsample         使用原始 1920x1088 分辨率跟踪
--openvins_root PATH    使用 PATH/install_ros2 替代默认构建
```

运行器会在隔离的 ROS 域中启动 OpenVINS，按时间戳顺序回放图像和 IMU。当视频解码滞后时，它会等待，避免突发发送 IMU 样本；同时记录轨迹，并在退出时关闭 ROS 进程。

### 输出与坐标系

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

`trajectory.npz` 是供后续集成使用的结果文件，包含相机和 IMU 的绝对时间戳、源帧索引、协方差、`T_world_imu` 和 `T_world_camera`。`T_world_camera` 表示左相机（`cam0`）位姿，由估计的 IMU 位姿与标定的刚体变换计算得到：

```text
T_world_camera0 = T_world_imu @ T_imu_camera0
p_world = T_world_camera0 @ p_camera0
```

OpenVINS 的世界坐标原点是局部且任意的。由于估计器使用 IMU，平移具有米制尺度；但结果并非全局配准地图，当前适配器也没有回环闭合功能。可通过相机时间戳与后续阶段关联；若两份结果来自同一会话，也可使用帧索引关联。

### Rerun 可视化

创建包含采样左相机画面的紧凑 Rerun 记录：

```bash
python \
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

rerun \
  --memory-limit 2GB \
  tmp/openvins_clip_0649_60s/trajectory/trajectory_with_left.rrd
```

轨迹会随时间轴推进逐步显示，亮黄色点表示当前左相机位置。全帧率轨迹仍保存在 `trajectory.npz` 中，Rerun 采样仅影响可视化。

当前适配器是离线参考实现：MP4 解码和 ROS 消息发布可能比估计器本身更慢。它尚未与手部、深度输出或 Kedro 默认流水线融合。验证、性能、长录制可视化及完整会话的详细说明见 [`docs/pipeline/openvins_vio.md`](docs/pipeline/openvins_vio.md)。

为使日志可复现，可以关闭 Kedro 遥测：

```bash
export KEDRO_DISABLE_TELEMETRY=1
```

## 流水线注册表

查看可用的 Kedro 流水线：

```bash
PYTHONPATH=. python - <<'PY'
from roboego_hand_vis.pipeline_registry import register_pipelines
for name in sorted(register_pipelines()):
    print(name)
PY
```

主要流水线：

```text
quality      原始数据完整性、视频基础质量、双目与深度质量
semantic     Qwen 帧标注、Stage1 动作、处理计划、手部窗口
datasets     片段导出、校正与清单准备、手部窗口导出
restoration  可选的 RealVDeblur 双目视频修复
depth        FoundationStereo 和 Fast-FoundationStereo
egohand      HaMeR、手部弱过滤、21 关节姿态门控、深度锚定、最终手部结果
objects      手物交互窗口的 SAM3 物体分割
privacy      前序处理完成后的会话视频人脸模糊
```

所有运行参数位于：

```text
conf/base/parameters.yml
```

大多数节点默认关闭。通过 `--params <section>.enabled=True,...` 仅启用需要的节点或阶段。

## 推荐的阶段启动器

日常处理推荐使用轻量级启动器，避免手动编写冗长的 `kedro run --params ...` 命令。数据集专用默认配置位于：

```text
conf/datasets/raw.yml
```

对一个会话运行单个阶段：

```bash
PYTHONPATH=. python \
  scripts/run_pipeline_stage.py \
  --dataset raw \
  --session /path/to/roboego-hand-vis/data/session_20260818_182102 \
  --stage raw_quality
```

`raw_quality` 同时执行完整性检查和质量门控。终端会输出门控状态、未通过的检查项和警告。

每次非预演（非 dry-run）调用都会写入或更新：

```text
<session>/pipeline_state.json
```

状态文件记录阶段状态、命令、耗时、返回码、生成的命令，以及工作流主动停止时的门控或阻塞原因。

对一个手部窗口运行单个阶段：

```bash
PYTHONPATH=. python \
  scripts/run_pipeline_stage.py \
  --dataset raw \
  --session /path/to/roboego-hand-vis/data/session_20260818_182102 \
  --stage depth_anchored_joints21 \
  --hand-window handwin_0000_t000000_000065
```

只打印生成的 Kedro 命令，不实际运行：

```bash
PYTHONPATH=. python \
  scripts/run_pipeline_stage.py \
  --dataset raw \
  --session /path/to/roboego-hand-vis/data/session_20260818_182102 \
  --stage final_hand_result \
  --all-hand-windows \
  --dry-run
```

对多个会话运行同一阶段，并生成静态 HTML 仪表盘：

```bash
python scripts/run_batch_pipeline.py \
  --root /mnt/nas/synnas/ego/baai_ego_task \
  --dataset raw \
  --stage raw_quality \
  --workers 4 \
  --dashboard
```

每个发现的会话都会生成本地状态文件：

```text
<session>/pipeline_state.json
```

状态文件记录阶段状态、命令、耗时、返回码、末尾输出和失败原因。`run_pipeline_stage.py` 的退出码 `2` 会被视为 `blocked`，通常表示门控主动停止了工作流。`base_prepare` 和 `phase1_depth` 等工作流别名是启动快捷方式，仪表盘会将其展开为实际阶段列。

仅生成或刷新仪表盘，不运行任何阶段：

```bash
python scripts/report_pipeline_state.py \
  --root /mnt/nas/synnas/ego/baai_ego_task \
  --dashboard
```

仪表盘默认输出路径：

```text
reports/pipeline_dashboard/index.html
```

首版支持的阶段：

```text
raw_quality
video_simple_quality
base_prepare
phase1_depth
post_annotation_actions
processing_plan
hand_windows
hand_visibility_gate
materialize_session
materialize_hand_windows
materialize_segments
materialize_segment_hand_windows
materialize_auto
prepare_stereo
stereo_quality
fast_foundation_depth
archive_depth_cache
hamer
hand_filter
joints21_pose_gate
depth_anchored_joints21
final_hand_result
final_hand_trajectory_polish
sam3_objects
video_privacy_blur
```

## 原始头带数据处理流程

为原始会话设置路径：

```bash
cd /path/to/roboego-hand-vis

export SESSION=/path/to/roboego-hand-vis/data/session_20260818_182102
export PLAN=$SESSION/annotation/processing_plan_left_v1.json
export HAND_WINDOWS=$SESSION/annotation/hand_estimation_windows_left_v1.json
export SEGMENTS_ROOT=$SESSION/segments
export STAGE1=$SESSION/annotation/qwen_frame_annotations_left_v4.checkpoint.json
```

如果终端未保留 shell 变量，请直接在 `--params` 中使用绝对路径。

### 1. 原始数据质量检查

检查相机视频、时间戳、时间戳对齐、视频帧率、IMU 时间戳和 IMU 信号质量。默认忽略四元数列。

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines quality \
  --to-nodes raw_headband_quality_gate \
  --params raw_headband_quality.enabled=True,raw_headband_quality.session_dir=$SESSION,raw_headband_quality.output_root=$SESSION,raw_headband_quality.overwrite=True,raw_headband_quality_gate.strict=False
```

输出：

```text
$SESSION/annotation/quality_report.json
```

关键字段：

```text
timestamp_alignment
timestamp_duplicates
imu.signal_quality
video_infos
```

### 2. 视频基础质量检查

在将帧发送给 VLM 之前，找出有效的视觉时间区间。

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session $SESSION \
  --stage video_simple_quality
```

输出：

```text
$SESSION/annotation/video_simple_quality_left.json
```

### 基础准备命令

在计算量较大的下游任务前，执行原始数据质量检查、视频基础质量检查、Qwen 秒级标注、标注后动作归一化、处理计划生成、手部窗口构建、视频裁剪与导出，以及双目准备。

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session "$SESSION" \
  --stage base_prepare
```

裁剪与导出步骤使用 `materialize_auto`：不足 3 分钟的会话直接准备，3–15 分钟的会话直接使用手部窗口，更长的会话则使用筛选片段加手部窗口。

### 手部可见性门控命令

在 `base_prepare` 之后、深度、手部、物体、VIO 或隐私等计算量较大的分支之前执行中间门控。

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session "$SESSION" \
  --stage hand_visibility_gate
```

手部可见性门控输出：

```text
$SESSION/annotation/hand_visibility_quality_gate_left_v1.json
```

门控根据 Qwen 的 `hands_visible.left/right` 为每个手部窗口计算 `core_both_hands_visible_rate`，默认阈值为 `0.9`。核心窗口前后的上下文时间段单独报告，不参与通过与否的判定。

### 3. Qwen 帧标注

对于手动裁剪的短会话，只有视频基础质量检查通过后才会运行 Qwen。默认沿原始视频时间轴标注，不按有效区间过滤。对于较长的原始视频，仍可通过 `qwen_frame_annotation.use_video_simple_quality=True` 启用有效窗口过滤。

该节点会写入批次输出、分块、检查点和最终合并的 JSON。多模态下载超时等远程 API 失败会在批次内重试。

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --to-nodes qwen_frame_annotation \
  --params raw_headband_quality.enabled=True,raw_headband_quality.session_dir=$SESSION,raw_headband_quality.output_root=$SESSION,raw_headband_quality.overwrite=True,raw_headband_quality_gate.strict=False,pre_semantic_quality.enabled=True,pre_semantic_quality.video=$SESSION/videos/left.mp4,pre_semantic_quality.output_json=$SESSION/annotation/video_simple_quality_left.json,qwen_frame_annotation.enabled=True,qwen_frame_annotation.video=$SESSION/videos/left.mp4,qwen_frame_annotation.output_json=$SESSION/annotation/qwen_frame_annotations_left_v4.json,qwen_frame_annotation.checkpoint_json=$SESSION/annotation/qwen_frame_annotations_left_v4.checkpoint.json,qwen_frame_annotation.raw_output_dir=$SESSION/annotation/qwen_frame_annotation_raw,qwen_frame_annotation.batch_output_dir=$SESSION/annotation/qwen_frame_annotation_batches,qwen_frame_annotation.chunk_output_dir=$SESSION/annotation/qwen_frame_annotation_chunks,qwen_frame_annotation.frames_dir=$SESSION/annotation/qwen_frame_annotation_frames,qwen_frame_annotation.prompt=docs/prompts/qwen_frame_annotation_v5_en.md,qwen_frame_annotation.fps=1,qwen_frame_annotation.require_video_simple_quality_pass=True,qwen_frame_annotation.use_video_simple_quality=False,qwen_frame_annotation.max_width=768,qwen_frame_annotation.batch_size=10,qwen_frame_annotation.batches_per_chunk=5,qwen_frame_annotation.model=qwen3.6-flash,qwen_frame_annotation.timeout_s=120,qwen_frame_annotation.max_retries=3,qwen_frame_annotation.retry_sleep_s=5,qwen_frame_annotation.resume=True
```

主要输出：

```text
$SESSION/annotation/qwen_frame_annotations_left_v4.checkpoint.json
$SESSION/annotation/qwen_frame_annotation_chunks/
$SESSION/annotation/qwen_frame_annotation_batches/
$SESSION/annotation/qwen_frame_annotation_raw/
```

### 4. 标注后动作处理

提取逐帧手部动作，构建文本复核分块，并消解动作词与物体绑定的差异。

推荐使用阶段启动器：

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session $SESSION \
  --stage post_annotation_actions
```



```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines semantic \
  --from-nodes stage1_action_extraction_chunks \
  --to-nodes stage1_action_text_resolver_chunks \
  --params qwen_frame_annotation.enabled=False,stage1_action_chunks.enabled=True,stage1_action_chunks.qwen_chunk_dir=$SESSION/annotation/qwen_frame_annotation_chunks,stage1_action_chunks.action_chunk_dir=$SESSION/annotation/stage1_action_chunks,stage1_action_chunks.review_chunk_dir=$SESSION/annotation/stage1_action_text_review_chunks,stage1_action_chunks.resolved_chunk_dir=$SESSION/annotation/stage1_action_resolved_chunks,stage1_action_chunks.model=qwen-plus,stage1_action_chunks.overwrite=False
```

输出：

```text
$SESSION/annotation/stage1_action_resolved_chunks/
```

### 5. 处理计划

根据 Qwen 标注和消解后的 Stage1 动作，生成待处理与丢弃的片段。`processing_plan.actions_json` 可以是单个 JSON 文件，也可以是消解后的分块目录。

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines semantic \
  --nodes build_processing_plan \
  --params processing_plan.enabled=True,processing_plan.stage1_json=$SESSION/annotation/qwen_frame_annotations_left_v4.checkpoint.json,processing_plan.actions_json=$SESSION/annotation/stage1_action_resolved_chunks,processing_plan.output_json=$PLAN,processing_plan.min_process_segment_s=5,processing_plan.context_expand_s=2
```

检查结果：

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

### 6. 手部估计窗口

根据处理计划构建聚焦手部的时间窗口。`stage1_actions` 也可以指定消解后的分块目录。

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines semantic \
  --nodes build_hand_estimation_windows \
  --params hand_estimation_windows.enabled=True,hand_estimation_windows.stage1_actions=$SESSION/annotation/stage1_action_resolved_chunks,hand_estimation_windows.processing_plan=$PLAN,hand_estimation_windows.output_json=$HAND_WINDOWS,hand_estimation_windows.hand_window_mode=two,hand_estimation_windows.context_expand_s=2,hand_estimation_windows.merge_gap_s=2,hand_estimation_windows.min_window_s=5,hand_estimation_windows.min_priority=high,hand_estimation_windows.fps=30
```

检查结果：

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

### 7. 视频裁剪与导出策略

处理计划决定哪些时间范围有价值；裁剪与导出（materialization）策略决定是否在计算量较大的双目、深度、手部和物体处理阶段之前，实际裁剪视频文件。

可选路径如下：

```text
A. 不裁剪
session -> prepare_stereo

B. 仅导出手部窗口
session -> hand_windows/handwin_xxx -> prepare_stereo

C. 先导出片段，再导出手部窗口
session -> segments/seg_xxx -> hand_windows/handwin_xxx -> prepare_stereo
```

推荐默认按会话时长自动选择：

```text
< 3 min      materialize_session
3-15 min     materialize_hand_windows
> 15 min     materialize_segment_hand_windows
```



```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session "$SESSION" \
  --stage materialize_auto
```

对于短数据或已手动预筛选的数据，跳过片段筛选，直接导出手部窗口：

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session "$SESSION" \
  --stage materialize_hand_windows
```

对于较长的原始会话，先导出筛选后的片段，再导出手部窗口：

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session "$SESSION" \
  --stage materialize_segment_hand_windows
```

如果整个会话已经是处理单元，则无需裁剪，直接将会话准备为双目处理单元：

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session "$SESSION" \
  --stage materialize_session
```

对一个或全部已导出的手部窗口执行双目准备；以下命令处理全部窗口：

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session "$SESSION" \
  --stage prepare_stereo \
  --all-hand-windows
```

查找当前手部窗口：

```bash
export HANDWIN_DIR=$(find "$SESSION/hand_windows" "$SESSION/segments" -path "*handwin_0000*" -type d | head -1)
export MANIFEST=$HANDWIN_DIR/dataset_manifest/dataset_manifest.json

echo "$HANDWIN_DIR"
echo "$MANIFEST"
```

## 双目质量与深度

在计算机器上执行双目准备、质量检查和深度处理。裁剪与导出策略按会话时长自动选择：

```text
< 3 min      session
3-15 min     hand_windows
> 15 min     segment_hand_windows
```



```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session "$SESSION" \
  --stage phase1_depth
```

深度估计前的双目质量检查：

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines quality \
  --nodes stereo_summary_quality \
  --params stereo_summary_quality.enabled=True,stereo_summary_quality.manifest=$MANIFEST,stereo_summary_quality.output_dir=$HANDWIN_DIR/quality/stereo_summary

/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines quality \
  --nodes stereo_segment_quality \
  --params stereo_segment_quality.enabled=True,stereo_segment_quality.manifest=$MANIFEST,stereo_segment_quality.output_dir=$HANDWIN_DIR/quality/stereo_segments,stereo_segment_quality.sample_every_s=1.0,stereo_segment_quality.orb_every_s=5.0
```

Fast-FoundationStereo 深度估计：

```bash
export DEPTH_DIR=$HANDWIN_DIR/depth/fast_foundation

/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines depth \
  --nodes fast_foundation_stereo_depth \
  --params fast_foundation_stereo_depth.enabled=True,fast_foundation_stereo_depth.manifest=$MANIFEST,fast_foundation_stereo_depth.output_dir=$DEPTH_DIR,fast_foundation_stereo_depth.start=0,fast_foundation_stereo_depth.num_frames=120,fast_foundation_stereo_depth.scale=0.5,fast_foundation_stereo_depth.valid_iters=4,fast_foundation_stereo_depth.max_disp=192,fast_foundation_stereo_depth.max_depth_m=4.0,fast_foundation_stereo_depth.overwrite=True
```

先使用 `num_frames=120` 估算处理速度。处理完整窗口时，请使用 `hand_estimation_windows_left_v1.json` 中的窗口帧数。

### 深度缓存归档

完整的 float32 深度图体积较大，应作为缓存管理。深度锚定使用完毕后，可将缓存归档为以毫米为单位的 uint16 PNG 帧，并可选生成 FFV1 无损 AVI：

```bash
PYTHONPATH=. python \
  scripts/run_pipeline_stage.py \
  --dataset raw \
  --session /path/to/roboego-hand-vis/data/session_20260818_182102 \
  --stage archive_depth_cache \
  --hand-window handwin_0000_t000000_000065
```

直接调用脚本：

```bash
PYTHONPATH=. python \
  scripts/depth/archive_depth_cache.py \
  --depth_dir $HANDWIN_DIR/depth/fast_foundation \
  --output_dir $HANDWIN_DIR/depth/fast_foundation_archive \
  --max_depth_m 4.0 \
  --fps 30
```

确认 `depth_uint16_ffv1.avi` 和 `depth_preview.mp4` 后，可使用以下命令在同一次运行中删除大型 float32 npy 缓存和临时 PNG 序列：

```bash
PYTHONPATH=. python \
  scripts/depth/archive_depth_cache.py \
  --depth_dir $HANDWIN_DIR/depth/fast_foundation \
  --output_dir $HANDWIN_DIR/depth/fast_foundation_archive \
  --max_depth_m 4.0 \
  --fps 30 \
  --delete_npy \
  --delete_png
```

通过启动器运行时，请在 `conf/datasets/raw.yml` 中设置 `archive_depth_cache.delete_npy` 和 `archive_depth_cache.delete_png`。

输出：

```text
$HANDWIN_DIR/depth/fast_foundation_archive/depth_uint16_png/
$HANDWIN_DIR/depth/fast_foundation_archive/depth_uint16_ffv1.avi
$HANDWIN_DIR/depth/fast_foundation_archive/depth_preview.mp4
$HANDWIN_DIR/depth/fast_foundation_archive/depth_archive_meta.json
```

编码后的深度单位为毫米。无效值、NaN、非正值和超出范围的值均写为 `0`。FFV1 AVI 用于机器可读的 16 位深度存储；`depth_preview.mp4` 是供人工快速检查的彩色视频。

## 第一人称手部流水线

对手部窗口运行 HaMeR：

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines egohand \
  --nodes hamer_windows \
  --params hamer_windows.enabled=True,hamer_windows.windows_json=$HAND_WINDOWS,hamer_windows.segments_root=$SEGMENTS_ROOT,hamer_windows.output_subdir=hand/hamer_raw,hamer_windows.hamer_root=/path/to/hamer,hamer_windows.stage1_actions=$STAGE1,hamer_windows.hamer_python=python,hamer_windows.body_detector=regnety,hamer_windows.batch_size=1,hamer_windows.wearer_roi_x_margin_ratio=0.0,hamer_windows.wearer_roi_y_margin_ratio=0.10,hamer_windows.save_mesh=True,hamer_windows.refine_mode=strict_ego_audit,hamer_windows.only_window_ids=handwin_0000,hamer_windows.overwrite=True
```

随后设置：

```bash
export HAMER_RUN=$HANDWIN_DIR/hand/hamer_raw
export REFINED_JSON=$HAMER_RUN/ego_hand_candidates_refined.json
export FILTER_DIR=$HANDWIN_DIR/hand/filtered
export JOINTS_GATE_DIR=$HANDWIN_DIR/hand/joints21_pose_gate
export ANCHORED_DIR=$HANDWIN_DIR/hand/joints21_depth_anchored
export FINAL_DIR=$HANDWIN_DIR/hand/final_hand_result
```

手部弱过滤：

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines egohand \
  --nodes hand_filter \
  --params hand_filter.enabled=True,hand_filter.manifest=$MANIFEST,hand_filter.refined_json=$REFINED_JSON,hand_filter.output_dir=$FILTER_DIR,hand_filter.start=0,hand_filter.num_frames=1980,hand_filter.max_bbox_jump_px=360,hand_filter.max_bbox_area_change_ratio=8.0,hand_filter.max_mano_pose_jump=0,hand_filter.max_mesh_centroid_jump=0,hand_filter.max_mesh_vertex_rmse=0,hand_filter.local_window_radius=2,hand_filter.save_vis=False
```

21 关节姿态门控：

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines egohand \
  --nodes joints21_pose_gate \
  --params joints21_pose_gate.enabled=True,joints21_pose_gate.final_tracks_json=$FILTER_DIR/filtered_hand_tracks.json,joints21_pose_gate.output_dir=$JOINTS_GATE_DIR,joints21_pose_gate.start=0,joints21_pose_gate.num_frames=1980,joints21_pose_gate.drop_fingertips_below_wrist=True,joints21_pose_gate.image_height=1050,joints21_pose_gate.max_suspicious_wrist_y_ratio=0.55,joints21_pose_gate.min_fingertip_below_wrist_ratio=1.0,joints21_pose_gate.min_fingertip_below_wrist_delta=0.02
```

21 关节深度锚定：

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines egohand \
  --nodes depth_anchored_joints21 \
  --params depth_anchored_joints21.enabled=True,depth_anchored_joints21.manifest=$MANIFEST,depth_anchored_joints21.candidates_json=$JOINTS_GATE_DIR/final_hand_candidates_gated.json,depth_anchored_joints21.depth_dir=$DEPTH_DIR,depth_anchored_joints21.output_dir=$ANCHORED_DIR,depth_anchored_joints21.start=0,depth_anchored_joints21.num_frames=1980,depth_anchored_joints21.anchor_mode=wrist_palm,depth_anchored_joints21.max_depth_m=2.5,depth_anchored_joints21.overlay_frames=auto
```

最终手部结果：

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines egohand \
  --nodes final_hand_result \
  --params final_hand_result.enabled=True,final_hand_result.tracks_json=$JOINTS_GATE_DIR/final_hand_candidates_gated.json,final_hand_result.output_dir=$FINAL_DIR,final_hand_result.depth_anchored_json=$ANCHORED_DIR/joints21_depth_anchored.json,final_hand_result.depth_anchored_npz=$ANCHORED_DIR/joints21_depth_anchored.npz,final_hand_result.depth_anchored_summary_json=$ANCHORED_DIR/joints21_depth_anchored_summary.json,final_hand_result.copy_meshes=True
```

最终轨迹优化：

```bash
/usr/bin/time -v env KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. \
python -m kedro run \
  --pipelines egohand \
  --nodes final_hand_trajectory_polish \
  --params final_hand_trajectory_polish.enabled=True,final_hand_trajectory_polish.joints_json=$FINAL_DIR/joints21_depth_anchored.json,final_hand_trajectory_polish.output_dir=$FINAL_DIR,final_hand_trajectory_polish.manifest=$MANIFEST,final_hand_trajectory_polish.max_interp_gap=10,final_hand_trajectory_polish.wrist_jump_m=0.0,final_hand_trajectory_polish.max_local_wrist_deviation_m=0.0,final_hand_trajectory_polish.local_window_radius=5,final_hand_trajectory_polish.smooth_window=9,final_hand_trajectory_polish.smooth_relative_fingers=True,final_hand_trajectory_polish.write_preview=True
```

## 物体分割

SAM3 仅处理从 Qwen/Stage1 标注中推断出的手物交互物体。

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session /path/to/session \
  --stage sam3_objects \
  --hand-window handwin_0000
```

输出：

```text
$HANDWIN_DIR/objects/<object_id>/clip_XXXX_tXXXXXX_XXXXXX/
```

SAM3 运行入口为本地脚本 `scripts/object/run_sam3_video.py`。如果当前环境未安装 SAM3 后端模块，请设置 `SAM3_PKG_DIR`。

## 会话视频隐私模糊

`privacy` 流水线的 `video_privacy_blur` 节点位于 `objects` 之后，作为最终视频隐私处理阶段。在默认 Kedro 图中，该节点依赖 `quality`、`semantic`、`datasets`、`restoration`、`depth`、`egohand` 和 `objects` 的完成报告，即使使用并行运行器，也会等待这些阶段全部完成后才执行。此约束不改变这些前序阶段之间原有的依赖关系。

节点仍默认关闭；运行完整默认流水线时，可在 `conf/base/parameters.yml` 中启用。单独调用具名 `privacy` 流水线或阶段启动器时，仍可直接处理已有会话，不会重新执行前序阶段。`base_prepare`、`phase1_depth` 和原 EgoBlur 节点保持原有行为。

实现代码已并入主项目的 `roboego_hand_vis/video_privacy_blur/`，测试位于 `tests/privacy/`，不再保留独立安装包或嵌套项目配置。通过主项目的可选依赖安装隐私处理所需库：

```bash
pip install -e ".[privacy]"
```

在 `conf/base/parameters.yml` 的 `video_privacy_blur` 中配置参数。相对路径均以本项目根目录为基准；默认 `python: python` 使用当前运行环境，也支持指定其他解释器路径。处理仍在子进程中执行，入口为 `roboego_hand_vis.video_privacy_blur.batch`，工作进程和资源探测进程使用同一项目包与解释器。仅注册 Kedro 节点不会加载模型库。

需要在 `PATH` 中提供 FFmpeg/ffprobe。项目已包含默认 YOLO 人脸权重 `conf/models/yolov8n-face.pt`（6,389,512 字节），节点和批处理 CLI 默认使用该项目内路径，也可通过 `face_weights` 覆盖。[来源与 SHA-256 记录](conf/models/yolov8n-face.pt.source.json)一并保存。Git 仅放行这份权重及其来源记录，其他模型仍保持忽略；默认配置无需另行下载模型。

处理单个会话：

```bash
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session /path/to/session \
  --stage video_privacy_blur
```

添加 `--dry-run` 仅打印 Kedro 命令，不处理视频或预热 GPU。启动器沿用 `pipeline_state.json` 记录状态；批处理入口和仪表盘通过现有机制支持新阶段。也可以直接调用节点：

```bash
python -m kedro run --pipelines privacy --nodes video_privacy_blur \
  --params 'video_privacy_blur.enabled=True,video_privacy_blur.session_dir=/path/to/session'
```

`camera_streams: auto` 会处理已存在的 `videos/left.mp4`、`right.mp4`、`bleft.mp4` 和 `bright.mp4`，忽略派生视频。可在 YAML 中指定 `[left, right]`，或在参数覆盖中使用 `left|right`。显式指定的相机缺失、或会话没有原始相机视频时会报错。

默认使用高斯模糊，检测置信度为 `0.35`、推理尺寸为 `960`、检测框扩展倍数为 `1.35`、模糊强度除数为 `1`，关闭时序补漏（`track_gap: 0`）。`method` 也支持 `pixelate` 和 `egoblur`。`jobs: 1` 限制单会话文件并发，设为 `auto` 可启用资源规划；`threads: 2`、`retries: 1` 和可选的 `timeout`（每个文件单次尝试的秒数，包含输出校验）控制资源使用。多会话运行时，总并发还受批处理启动器的 worker 数影响。

默认输出：

```text
<session>/privacy/video_privacy_blur/
├── left_blurred.mp4
├── right_blurred.mp4
├── session_report.json
├── run.log
└── .batch/
    ├── jobs/
    ├── logs/
    ├── previous/
    └── summary.json
```

输出使用“输入文件主名 + `_blurred.mp4`”：例如 `left.mp4` 对应 `left_blurred.mp4`，`right.mp4` 对应 `right_blurred.mp4`，并保留相对目录层级。同一目录下的 `a.mp4`、`a.mov` 会产生同名输出，因此提前报错。已有的 `left.mp4.blurred.mp4` 等旧命名文件会保留；更新后首次运行生成新命名文件，后续按新文件进行断点恢复。

会话报告记录相机输入输出映射、计数、失败原因和日志路径。单文件失败后继续处理其他文件，存在失败时节点与阶段最终失败。请保留 `.batch/` 用于恢复；重复运行会核验输入、配置、模型、源码及运行环境签名和输出，匹配后才跳过成功文件。不会覆盖没有任务记录的已有输出。Ctrl-C 或 SIGTERM 会转发到批处理进程并清理子进程。

后端测试已纳入主项目测试集。安装隐私与开发依赖后，可以运行相关测试：

```bash
pip install -e ".[dev,privacy]"
PYTHONPATH=. python -m pytest tests/privacy tests/pipelines/test_video_privacy_blur.py -q
```

原视频、标定、传感器时间戳和下游清单均不修改。输出为无音轨的 H.264 MP4，保留原帧数、尺寸和恒定帧间隔，时间戳从零开始。后端拒绝变帧率、旋转元数据和奇数尺寸视频。文件校验仅验证视频完整性，不代表人脸检测覆盖率通过验收；分享前请检查成片。

## 可选节点

RealVDeblur 双目视频修复：

```bash
kedro run --pipelines restoration --nodes realvdeblur_stereo --params ...
```

旧版网格与深度对齐流程仍然可用，但已不再是推荐主流程：

```text
legacy_mesh_depth_alignment
legacy_track_fill_smooth
legacy_export_joints21
```

推荐的手部与深度处理路径：

```text
hamer_windows
hand_filter
joints21_pose_gate
depth_anchored_joints21
final_hand_result
final_hand_trajectory_polish
```

## 外部已处理数据集

对于 Ropedia 等已处理的双目数据集，先解析生成 `dataset_manifest.json`：

```bash
PYTHONPATH=. python scripts/datasets/resolve_dataset.py \
  --dataset ropedia \
  --root /path/to/ropedia_data \
  --output_dir tmp/ropedia_dataset_manifest
```

随后使用生成的清单运行相同的 quality、depth、egohand 和 objects 阶段。

## 开发

运行测试：

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib -q
```

运行部分测试：

```bash
PYTHONPATH=. python -m pytest tests/pipelines -q
```

不要提交生成的数据、视频、模型输出、掩码、网格和 `.rrd` 文件。大型输出应保存在 `data/`、`tmp/` 或外部存储中。
