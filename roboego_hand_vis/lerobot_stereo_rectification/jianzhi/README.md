# 简智双目去畸变与立体矫正

已有结果复用时只检查视频可打开、尺寸、FPS、容器声明帧数与时间戳及参数一致性，不再逐帧完整解码。首次处理仍必须逐帧解码以完成重映射，但完成后不再额外解码一遍。

正式去畸变前，在原始 `calibrations/camera-camchain.yaml` 顶层追加 `head_local_to_calibration_body`，记录 `p_body = R @ p_head_local + t`：R 为 `[[0,1,0],[0,0,-1],[-1,0,0]]`，t 为零（原点重合假设）。六个原始相机条目保持不变，不新建坐标关系文件。该关系用于手部 world 点经逐帧头部/VIO 逆变换后转入标定 body 系；这里只记录，不修改手部点或校正算法。记录标记为 inferred，依据是样本 001301... 前 5 秒的人工腕点及手指叠加验证，未确认适用于其他样本或明显头动。重复运行不重复写入，已有不同关系时报错；dry-run 不写文件。

正式处理前也会删除 `meta/jianzhi_quality_report.json`（如果存在）；`--dry-run` 保留该文件。

处理 `observation.images.head_left`、`observation.images.head_right` 前向双目，复用 L1 的双球模型重映射、共同有效区域裁切、视频编码和输出校验。外侧四目不参与此流程。

单条数据（仓库根目录执行）：

```powershell
python -m roboego_hand_vis.lerobot_stereo_rectification.jianzhi.run `
  --source D:\data\jianzhi\001301e26b8c4ace844c7b2d12eacf9d `
  --workers 1 --ffmpeg C:\path\to\ffmpeg.exe
```

服务器批次：

```bash
python -m roboego_hand_vis.lerobot_stereo_rectification.jianzhi.run \
  --source /share/project/liuyou/humancoin/data/jianzhi/batch_0000001 \
  --workers 4 --ffmpeg /share/project/liuyou/humancoin/envs/ego/bin/ffmpeg
```

`--source` 可以是单条数据、批次或包含多个批次的目录。`--dry-run` 仅检查输入及输出计划；`--no-skip-existing` 强制重算，默认验证已有结果并复用。`--sessions-file` 接受 source 内选定数据集的 JSON 路径数组。需要 numpy、opencv-python、pyarrow、PyYAML 和支持 libx264 的 ffmpeg。

正式处理前删除 `calibrations/jianzhi-camera-calibration.json`（如果存在），标定只读取 `calibrations/camera-camchain.yaml`，不回退到 JSON。`--dry-run` 也读取 YAML，但不删除文件。简智的 `camera_model=pinhole, distortion_model=ds` 实际按双球模型解释，D 顺序为 `[fx, fy, cx, cy, xi, alpha]`，适配为引擎使用的 `[xi, alpha, fx, fy, cx, cy]`。外参约定为 `T_b_c` 将相机坐标变换到 body，故左右目变换为 `inverse(T_b_right) @ T_b_left`。校验旋转、基线、参数一致性和输出正视差方向。

默认视频尺寸必须与标定一致，不推测缩放。只有明确确认视频为居中加边时才使用 `--center-crop-calibration`。`--balance` 范围为 0～1，默认 0；增大可扩大虚拟视野，最终仍按共同有效区域裁切。

在原数据集新增：

- `videos/chunk-*/observation.images.head_{left,right}_rectified/episode_*.mp4`
- `data/chunk-*/episode_*/observation.images.head_{left,right}_rectified.parquet`：原始时间戳的副本。
- `calibrations/rectify_params.yaml`：R1/R2/P1/P2/Q、输出内参、基线对应变换及裁切信息。
- `calibrations/dataset_manifest.json`：标定来源、视频及内参；多 episode 时文件名带 chunk/episode 前缀。不生成 frame_mapping.parquet；按源帧序处理，不重新配对或重采样。

同时在 `meta/info.json` 注册新增视频特征并更新视频总数。原视频、标定和 master_index 保留；每目逐帧处理，不执行时间重采样或双目同步。

样例验证：从 `001301e26b8c4ace844c7b2d12eacf9d` 前向双目各截取 12 帧，在临时副本运行，输出 1600×1300、30 FPS，逐帧解码数量及时间戳一致，复用校验通过。原始 5402 帧视频未全量运行。首帧 SIFT 匹配的 RANSAC 内点中约 99.8% 为正视差，垂直残差中位数约 5.8 像素；这是样例观测，不能视为深度质量验收，仍需核查原始标定及时间同步。

2026-11-06 复查 D 盘样本：左右视频各 5402 帧，标定和视频均为 1600×1300，基线约 0.059404 m，无输入及输出裁剪。独立 12 帧副本的编码、逐帧解码、时间戳内容相等与结果复用检查均通过；4 项标定适配测试通过。原始数据未修改。

在第 0、1350、2700、4050、5390 帧，使用 SIFT 比率筛选（0.7）和基础矩阵 RANSAC（1 px）筛选匹配后，垂直坐标差绝对值中位数分别为 5.68、5.54、6.05、5.78、5.64 px，95 分位为 6.70～8.26 px；入选点均为正视差。RANSAC 在此仅剔除错误匹配，并不保证水平极线对齐。这些结果表明仍有可测量的垂直残差，不能仅凭处理成功认定立体校正精度达标，也不足以单独归因于标定或同步。

本机复查报告与极线对照图保存于 `D:\data\jianzhi_rectification_review\review.json` 和 `rectified_epilines.jpg`，12 帧输出在该目录的 `sample_12_frames/`。
