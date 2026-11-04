# LeRobot 头环基础质检

独立检查 L1 转换器生成的 Multi-Frequency LeRobot 数据。原有
`raw_headband_integrity` 不变。仅写入 `annotation/quality_report.json`，
不修改视频、Parquet、WAV、标定或数据集元数据，不自动修复文件。
每次执行重新检查并更新报告；不依赖数据库或 GPU。

## 执行

在 roboego-hand-vis 项目根目录运行，`--source` 支持单个数据集或批次目录：

```bash
python -m roboego_hand_vis.lerobot_headband_integrity.L1.run \
  --source /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001 \
  --workers 4
```

也可以直接执行 `roboego_hand_vis/lerobot_headband_integrity/L1/run.py`。
`--workers` 控制 CPU 工作进程数，每个进程处理一个 session。
`--dry-run` 只列出数据集，不写报告；隐藏的失败转换临时目录不参与扫描。
退出码 0 表示全部通过（允许警告），1 表示有失败，2 表示参数错误。

额外依赖：`pip install pyarrow PyYAML`；`ffprobe` 需在 PATH 中，
也可用 `--ffprobe /path/to/ffprobe` 指定。不需要安装 LeRobot、Torch 或 OpenCV。

## 检查内容

| 模块 | 功能 |
| --- | --- |
| `video.py` | ffprobe 完整解码计数、错误信息、尺寸、帧数与时间戳及主索引一致；FPS 差异警告 |
| `timestamps.py` | 相机时间戳非空、有限、严格递增、异常间隔、主索引匹配；按最近时刻比较相机 |
| `imu.py` | 六维数据有效性、时间戳、全信号全零、覆盖首尾及内部缺口；冻结和突跳警告 |
| `audio.py` | 只读检查 WAV 声明长度与实际可读数据、音视频时长差 |
| `calibration.py` | 标准相机齐全、内参、畸变、尺寸、存在的外参矩阵及链式参考名称 |
| `compute.py` | 模态注册、异常隔离、报告和门禁结果 |
| `common.py` | 新格式路径、Parquet 读取和统一检查结果 |
| `run.py` | 扫描及多进程调度 |

遵循四路特征：`head_left`、`head_right`、`head_left_outer0`、
`head_right_outer0`。路径适配项目的
`data/chunk-*/episode_*/*.parquet` 和
`videos/chunk-*/observation.images.head_*/episode_*.mp4` 布局。
使用各流真实时间戳；不以同序号帧代替同时刻帧。

默认最近匹配容差 0.02 秒，重叠时间段内超过容差的比例大于 0.001 时失败；
首尾不覆盖单独警告，完全无重叠失败。相机间隔大于 2.5 帧时警告。
IMU 单轴全零不直接失败（静止时可能出现），六轴全部为零失败。
IMU 覆盖缺口最多 10 个相机帧，音视频时长差最多 5 秒。

两份 VIO 标定 `camera_imu-imu.yaml`、`camera_imu-camchain-imucam.yaml` 的结构完整性为失败门禁。
启用 `--repair-calibration-root` 时先从同 cp 编号、QC 通过的合格供体补齐，再复检；
找不到供体或复检不通过，该 session 不放行到任何后续阶段。相机间 nearest_alignment 超限仍仅警告。
可通过 `--timestamp-tolerance-s`、`--imu-gap-frames`、
`--audio-duration-tolerance-s` 调整。时长检查不能证明音视频同步。
多个 episode 没有 WAV 分段映射时，音频只检查可读性并给出警告。
标定检查验证结构，不验证实际物理标定精度。

报告包含 `passed`、`gate.failed_checks`、`gate.warnings` 和逐项 `checks`。
新增模态只需新增一个含 `check(context)` 的模块，并注册到
`compute.MODALITIES`；每项返回 `common.result(...)`。


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

Batch pipeline shares an immutable calibration donor snapshot per invocation (--calibration-donor-cache). Discovery scans only batch/session annotation directories, without descending into videos or intermediate outputs.

## First-frame horizontal stereo alignment

Requires OpenCV with SIFT and NumPy. For each episode, read frame zero from head_left and head_right. Load L1 calibration from calibrations/camera-camchain.yaml, then rectify in memory using the same center crop, balance=0 and common valid ROI as stereo rectification. No precomputed rectified videos are required. L3 separately loads its episode calibration from meta/calibrations.jsonl.

Both levels use the depth gate metrics: at least 30 mutual SIFT matches, at least 4 occupied cells in a 4-by-3 grid, median vertical error <=20 pixels and P95 <=40 pixels. Excessive error, insufficient matches, invalid calibration or unreadable first frames fail QC. Metrics appear under stereo_first_frame/<chunk>/<episode>/horizontal_alignment in quality_report.json. Only frame zero is assessed.
