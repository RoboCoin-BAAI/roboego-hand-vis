> Default: `--init-mode static-then-dynamic`. Other policies: `static-zupt`, `dynamic`, `dynamic-then-static`. See [initialization policy](INITIALIZATION.md). The ZUPT flags below are retained for legacy commands.

# LeRobot VIO

- [L3 数据入口](L3/README.md)：头环左右目、headband_imu0 与 JSONL 标定加载。
- 下文为 L1 入口及共用算法、输出说明。

## 常规 VIO 优先，ZUPT 失败重试

批量命令用 `--zupt-fallback` 替换 `--zupt`。第一次运行仍是双目+IMU VIO，
只关闭 ZUPT，并非纯视觉里程计。通过基本质量检查就结束；质量失败或后端
非零退出时启用 ZUPT 重试一次。输入、标定、结构或发布错误不通过切换算法重试。
`--zupt` 和 `--zupt-fallback` 互斥，`--prepare-only` 不触发重试。

常规和 ZUPT 报告分别保存在 `vio/`、`vio_zupt/`；规范字段沿用已完成的发布逻辑，
常规通过时不再被 ZUPT 覆盖，重试通过后发布 ZUPT 结果。两次都失败时仍标记失败，
不会把结果认定为合格；失败轨迹按既有逻辑保留并带质量标记，使用时需过滤。
常规输出目录中的 `fallback_selection.json` 记录两次状态和最终选择，终端显示
`selected_mode=normal/zupt`，都失败为 null。既有旧 ZUPT 报告不会被删除，
是否是本次选中结果以 selection 和 episode 内 provenance 为准。

## 可选 ZUPT 对照

原命令增加 `--zupt`，只将生成配置中的 `try_zupt` 改为 true，默认关闭。
保留现有 `zupt_*` 设置（chi2 multiplier=0、noise multiplier=10、max velocity=0.1、
max disparity=0.5、only at beginning=false），不修改原标定或估计器判定逻辑。
现有实现允许图像视差检查覆盖惯性检验，不能把“触发”理解成独立确认静止；
必须结合视频检查缓慢头部运动是否误触发。

开启时矫正模式输出到 `vio_zupt/chunk-XXX/episode_XXXXXX/`，raw 模式输出到
`vio_raw_zupt/`，不覆盖原来的 `vio/`、`vio_raw/` 报告目录。
矫正模式（含 `--zupt`）直接更新规范 LeRobot 位姿、状态特征和元数据，覆盖该
episode 上次发布的位姿。raw 模式仍只保存独立对照结果。
episode 内 `relative_head_pose_provenance.json` 记录当前字段对应的报告路径和
ZUPT 设置，便于区分 `vio/` 与 `vio_zupt/` 的结果来源。
NPZ、质量检查、IMU 对照仍按原流程生成。默认关闭时保持原有输出行为。

同步两个仓库并重编译 OpenVINS 后，`estimator_diagnostics.csv` 新增
zupt_considered、zupt_attempted、zupt_applied；汇总 JSON 的 `zupt` 包含实际
try_update 次数、触发帧和时间。applied 可包括同时间戳复用的判定，不能直接
等同于新增 EKF 更新调用。旧二进制没有触发记录，trace_available=false，
此时零次记录不代表未触发。ZUPT 帧不属于视觉更新，无视觉更新比例需结合
ZUPT 触发情况解读。开启不会自动把质量失败改为通过，也不保证抑制发散。

## 估计器内部诊断

同步本项目和 open_vins 的诊断改动，重新编译原离线目标后，原命令自动输出
`estimator_diagnostics.csv` 和 `estimator_diagnostics.json` 到对应 `vio/` 或
`vio_raw/` 的 episode 目录。旧二进制仍可运行，但不会有这两份新诊断。
JSON 同时嵌入 report.json，包含初始化后前十帧速度/零偏/预测重力方向、
特征拒绝统计、最长无视觉更新时长，以及视觉更新前后位置改变量最大的十帧。
CSV 提供逐帧细节；pre/post 区分惯性传播与相机更新阶段的改变量。
计数是更新尝试数，不是互不重复的特征总数；姿态/零偏是估计值，不能视为真值。
本改动不调整噪声、阈值、ZUPT、初始化或轨迹输出，不自动修复发散。

扩展诊断需同步 `open_vins/ov_core`、`ov_msckf`、`ov_offline` 后重新编译。
新增 `feature_diagnostics.csv` 记录逐特征拒绝原因、深度、条件数、基线比例、
投影前的像素坐标残差 RMS 和实际卡方检验值/阈值。未执行的检查为 NaN。
`estimator_diagnostics.json` 中 `feature_details` 汇总各阶段原因与最初12个
候选更新帧的残差。逐帧状态还包含视觉更新前后零偏、旋转与协方差对角线，
协方差是方差而不是标准差。SLAM 新点初始化的内部卡方数值未暴露，保持缺失。

## 原始鱼眼视频对照

使用 `--input-mode raw` 读取 `observation.images.head_left/head_right` 的视频和
Parquet 时间戳，使用原始相机—IMU 标定，不要求已有立体矫正产物。
raw 输出到 `SESSION/vio_raw/chunk-XXX/episode_XXXXXX/`，包含原后端坐标系
下的 trajectory.npz、质量报告和 IMU 对照，暂不写入规范位姿 Parquet 或更新
LeRobot 元数据，以保留 rectified 结果做对比。

```bash
python -m roboego_hand_vis.lerobot_vio.L1.run \
  --source /data/L1_lerobot/batch_0000001/cp_0001_session_20261009_172508 \
  --openvins-root /code/open_vins --workers 1 \
  --input-mode raw --raw-center-crop-calibration \
  --time-offset-mode estimate --timing-bootstrap auto
```

尺寸一致时直接读取原视频。尺寸不一致时默认报错：只有确认原标定描述的是
完整视频中央 ROI，才添加 `--raw-center-crop-calibration`。例如 1920×1200
裁为 1920×1088，上下各去掉 56 像素，与现有矫正器的中心裁剪假设一致；
不缩放、不去畸变、不改标定内参。使用 OpenCV FFV1 编码生成本次 .work 内
无损裁剪 MKV，保留所有帧和采集时间戳。需要 OpenCV 支持 FFV1，且会增加
准备时间和磁盘占用。如果实际是缩放或非中心裁剪，不能使用这个假设。
raw 和 rectified 应以相同下采样及时间偏移参数做对比。

使用原有 VIO 标定转换与 `open_vins` 的 ROS-free 离线后端。默认读取
`head_left_rectified` / `head_right_rectified`，需先完成 L1 立体矫正。
不依赖深度估计，不修改原始视频、传感器 Parquet 或标定。
运行后写入派生位姿 Parquet，并更新 LeRobot 的特征定义和逐 episode 统计。

运行环境需要 Python 3.10+、numpy、opencv-python、PyYAML、pyarrow；
OpenVINS 使用 `fix/vio-cold-start` 分支，并编译 `roboego_openvins_offline` 目标。
`--openvins-root` 默认是当前项目旁边的 `open_vins`，使用默认的
`ov_msckf/build/openvins_offline_native` 构建位置。

在 roboego-hand-vis 根目录运行（Linux）：

```bash
python -m roboego_hand_vis.lerobot_vio.L1.run \
  --source /data/L1-lerobot/batch_0000001/cp_0001_session_20261009_172052 \
  --openvins-root /code/open_vins

# 整个 batch；batch 名称没有固定的位数要求
python -m roboego_hand_vis.lerobot_vio.L1.run \
  --source /data/L1-lerobot/batch_000001 \
  --openvins-root /code/open_vins --workers 2
```

也可直接执行 `roboego_hand_vis/lerobot_vio/L1/run.py`。
`--dry-run` 只列出路径并检查文件存在，不需要已编译的二进制、不写入结果；
`--prepare-only` 校验视频、读取传感器 Parquet 并生成输入，不运行 OpenVINS。
`--start-s` / `--duration-s` 设置相对第一个双目配对时间的运行区间。
`--stereo-tolerance-s` 默认 0.001 秒；矫正图像默认不下采样。
`--time-offset-mode estimate`、`--timing-bootstrap auto` 与现有服务器入口默认值一致。
要遵循 YAML 时间偏移设置可用 `--time-offset-mode config --timing-bootstrap off`。

每个 episode 输出到 `SESSION/vio/chunk-XXX/episode_XXXXXX/`：

- `trajectory.npz`：后端直接导出的 `openvins.frame_aligned.v2`，按左视频原帧号对齐。
- `report.json`：后端运行报告、LeRobot 来源信息及基本质量校验。
- `.work/<run-id>/`：本次输入、后端详细报告和 `console.log`。

NPZ 的 `status == 'valid'` 才是有效位姿；无位姿帧保留 NaN，不插值或补齐。
相机与 IMU 都使用 Parquet 内的原始 `timestamp` 秒值，保持共同时间原点，
不分别归零，不按 FPS 推算，不手工再叠加标定时间偏移。
IMU 从 gx/gy/gz 与 ax/ay/az 映射为角速度 rad/s 和加速度 m/s²。
IMU 边界覆盖和内部缺口检查由 OpenVINS 离线后端执行。

批次内单个任务失败后继续其他任务，最终返回非零退出码。
`--workers` 并行处理不同 session，同一 session 的 episode 顺序处理，防止元数据覆盖。
后端失败或结构校验失败保留旧结果及本次诊断文件；
旧结果存在不代表本次运行成功。不支持多个进程同时处理同一个 episode。
完整 VIO 运行须在有已编译 OpenVINS 的 Linux 环境验证；Windows 可检查路径和准备输入。

## LeRobot 位姿字段

直接写入 `data/chunk-XXX/episode_XXXXXX/`：

- `observation.relative_head_pose.parquet`：timestamp、episode_index、x/y/z/qx/qy/qz/qw。
- `observation.relative_head_pose_status.parquet`：timestamp、episode_index、status（整数）。

遵循运动特征 `observation.<name>` 命名。没有全局世界对齐，所以不占用
`observation.head_pose`。位姿是 `T_first_valid_imu_current_imu`：以本段首个有效
IMU 为参考，平移单位米，旋转为 Hamilton xyzw。IMU 固连于头部设备，
没有假设额外的人体头部坐标系变换。每段参考原点独立，不能直接拼接。
无效帧保留 NaN，通过配套状态字段判断；不插值，不补零。
状态码在 `meta/info.json` 对应 feature 的 `status_codes` 中记录。
原视频帧序、时间戳、master_index 保持不变。
`meta/episodes_stats.jsonl` 添加新字段统计，位姿统计仅使用有效帧。

## 基本质量检查

检查帧号、时间戳、数组形状、有效位姿有限性、齐次矩阵末行、旋转正交性与行列式。
结构损坏阻止发布，并在本次 `.work/<run-id>/quality.json` 写入原因。
以下阈值可通过命令行调整（默认值是可调整的工程筛查值，不是精度标准）：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--min-valid-ratio` | 0.5 | 有效位姿占完整左视频帧数的比例，包括初始化/无 IMU/未请求帧 |
| `--max-step-m` | 1 | 相邻有效位姿的平移距离上限，含跨缺失帧间隔 |
| `--max-speed-m-s` | 5 | 使用真实时间差计算的平均平移速度上限 |
| `--max-angular-speed-deg-s` | 720 | 相邻有效位姿平均角速度上限 |
| `--max-distance-m` | 50 | 距首个有效位置的最大距离，提示可能漂移 |

报告包含有效比例、首尾有效帧、轨迹长度、首尾位移、最大距离、速度和跳变帧号。
超过阈值仍保存可追查的 LeRobot 结果，但标记 `quality_status=failed`，
命令返回非零。通过时为 `passed_basic_checks`，不表示精度已验证。
没有真值/闭环约束时不能测量真实漂移；大范围正常运动也可能触发最大距离阈值。
使用短时间窗口时可按需要调低比例阈值，因为分母仍是完整视频帧数。

## IMU 与 VIO 对照

无需新增参数，原命令自动执行：

- 运行前生成 `imu_diagnostics.json`：采样率、最大间隔、六轴均值、加速度/角速度
  模长、异常大值、六轴完全固定的持续时间，以及按初始标定偏移计算的视频边界覆盖。
- 运行后生成 `imu_vio_comparison.json`：使用每帧 `timestamp_imu` 对齐原始 IMU，
  比较相邻有效视频帧的 VIO 净转角与陀螺仪模长积分；后者是转动量上界，
  不视为精确姿态。记录净转角超出积分 0.05 rad 的区间。
- 对持续约 0.5 秒低角速度、低加速度波动的区间，记录 VIO 速度超过 1 m/s
  的帧，同时列出既有质量检查标记的跳变/超速事件及附近 IMU 大值和间隔情况。
- 记录在线时间偏移的初值、终值与变化量。两份诊断同时嵌入 `report.json`。

诊断文件保存在 `vio/chunk-XXX/episode_XXXXXX/`，运行前诊断还保留在本次
`.work/<run-id>/`，即使后端失败也可检查。`--prepare-only` 也会生成运行前诊断。
对照不跨越缺失 VIO 帧，不外推 IMU，不跨越超过 50 ms 的 IMU 缺口。
这些是原始 IMU 的工程诊断，未应用滤波器内部陀螺仪零偏/尺度校正；
低 IMU 变化既可能是静止，也可能是匀速。对照警告不自动改变质量门禁、不证明
IMU 故障，也不生成修正后的位姿。20–60 秒视频仍可能迅速发散，
因此不假设“短视频必然无漂移”，也不将 IMU 双积分当位置真值。
