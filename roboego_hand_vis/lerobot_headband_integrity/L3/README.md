# L3 LeRobot 基础质检

支持单个 session、batch 目录或整个 L3_lerobot 目录，递归发现 meta/info.json。
写入各 session 的 annotation/quality_report.json。
先检查四路头环图像（bright、left、right、bleft）的声明、视频和时间戳文件。
图像文件齐全而 meta/calibrations.jsonl 缺失时，自动从匹配供体补齐标定。
已有标定即使不合格也不会覆盖；无合格供体则标定检查失败。
供体需目录 L3 编号及 collect_info.jsonl 的 collector_id 相同，hardware.ego_headband 型号相同，
且标定结构检查通过。优先同 batch，再按 session 时间接近程度选择；
不使用带 l3_calibration_repair.json 的已修复 session 作为供体。
仅复制标定内容，按目标 episode 重建 episode_index、session，来源记录于
annotation/l3_calibration_repair.json。原数据中的其他文件不修改。
每次重新检查，视频使用 ffprobe 完整解码计数，不使用 L1 的缓存。

```bash
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_headband_integrity/L3/run.py \
  --source /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001 \
  --calibration-root /share/project/liuyou/humancoin/data/L3_lerobot \
  --workers 4
```

也支持项目根目录执行 `python -m roboego_hand_vis.lerobot_headband_integrity.L3.run`。
添加 `--dry-run` 只列出数据集；`--sessions-file` 接受所选 session 绝对路径的 JSON 列表。
--calibration-root 指定跨 batch 的供体搜索目录；未指定时 batch 输入默认搜索其父目录。
依赖 Python 3.10+、pyarrow 和 PATH 中的 ffprobe，可用 --ffprobe 指定路径。
退出码：0 全部通过（允许警告），1 存在失败，2 参数或数据集选择错误。

- 按 info.json 检查八路相机声明（含四路头环）、episode 数、主索引总帧数。
- 传感器仅检查头环 observation.headband_imu* 的 Parquet、数值列宽度、列名和数值有限性；全零记 warning。
  IMU 跳变沿用 L1：某列最大相邻差值 ≥5，且相对非零相邻差值中位数 ≥50 倍
  （中位数为零时同样视为异常），判失败并记录 jump_columns。
  跳过手腕 IMU、眼动 IMU、EEG、EMG、眼动、姿态、action，不读取其 Parquet。
  不检查时间戳非空、有限、递增、对齐或首尾覆盖，也不产生相关 warning。
- 视频检查解码错误、真实帧数与本路时间戳条数、尺寸、FPS。
- WAV 检查实际数据长度、采样率、Parquet num_samples 总和及主索引时长。
- 标定读取 meta/calibrations.jsonl，检查 episode 对应、ego_headband 三组标定、四路标定相机、内参和存在/必需的外参矩阵。

各路保留原生频率，不要求与主索引帧数相等；不把 left_wrist_rgb 与 L1 head_left 混用。
全零信号与视频 FPS 偏差为警告。
音频时长容差默认 5 秒，可用 --audio-duration-tolerance-s 调整。

基础质检不判定 EEG/EMG 生理有效性、阻抗是否合格或物理标定精度；
meta/alignment_check.jsonl 是转换记录，本质检不再复查时间戳对齐。
标定分辨率不与视频直接强制相等：当前样本标定为 1920×1088，头环视频声明为 1920×1200，
需要明确采集与转换的裁剪关系后才能用于几何处理。


## Strict QC rules

- Require observation.headband_imu0 (six-axis float data) and its parquet in each episode.
- Check all video/audio/headband-IMU timestamps and the master timeline. Empty or
  nonfinite timestamps fail. Finite repeated/backward IMU timestamps are warnings;
  finite video/master non-increasing timestamps warn; audio non-increasing timestamps fail.
- Accept both pinhole/equidistant and ds/none camera calibration. DS accepts
  negative xi, six intrinsics, and empty distortion coefficients.
- Validate proper rigid transforms (orthonormal rotation, determinant +1),
  required camera-chain links, camera/IMU time offsets, and positive finite IMU
  noise densities, random walks, and update rate. Donor calibration uses the same checks.


IMU signal policy (L1 and L3): inspect each axis for near-constant samples.
Any axis with exactly identical values across all frames (at least two samples)
fails QC. Near-freezing of any single axis also fails: at most 0.1% of adjacent
differences exceed 1e-12. Reports include constant_columns.
A jump fails only when the adjacent difference is at least 20 in the stored axis
units AND at least 100 times the median nonzero adjacent difference (a zero
baseline also triggers the relative condition). Previous thresholds were 5 and 50.
Reports retain affected axes and effective jump thresholds. These are heuristic
signal checks, not proof of hardware failure.

Finite repeated/backward video and master timestamps are warnings. Empty or nonfinite timestamps still fail. VIO independently requires strictly increasing camera timestamps.

Front headband stereo boundary QC: absolute first-frame and last-frame timestamp differences, multiplied by left-camera FPS, must each be <= 2 frames. Either exceeding 2 fails QC. Uses actual frame order, without sorting timestamps.

左右水平对齐质检：每个 episode 只读取 headband_left_rgb / headband_right_rgb 的首帧，使用与 L3 双目校正、深度输入相同的标定、居中裁剪、balance=0 和公共有效区域裁剪，在内存中完成去畸变和双目校正，无需先生成完整校正视频。复用深度模块的 SIFT 双向特征匹配指标：至少 30 对匹配、覆盖至少 4 个网格，垂直误差中位数不超过 20 像素、P95 不超过 40 像素。指标和标定来源写入 quality_report.json 的 stereo_frame_10/horizontal_alignment 检查；超限、匹配不足或首帧/标定不可读均不通过。此检查仅覆盖首帧。

L3 QC ?????? 10 ??frame_index=9??P95 ????????????????max_p95_px=null??????? 30 ???????? 4 ?????????? ?20px?L1 ?????????????
