> Default: `--init-mode static-then-dynamic`. Other policies: `static-zupt`, `dynamic`, `dynamic-then-static`. See [initialization policy](../INITIALIZATION.md). The ZUPT flags below are retained for legacy commands.

L3 rectified runs publish `data/<chunk>/<episode>/observation.left_camera_pose.parquet`
and `observation.left_camera_pose_status.parquet`. Pose columns are x, y, z
(meters), qx, qy, qz, qw (Hamilton). Poses use the rectified left camera frame,
relative to the episode's first valid left camera pose. OpenVINS supplies
`T_world_camera = T_world_imu @ T_imu_camera`; the stored relative pose is
`inverse(T_world_camera_first) @ T_world_camera_current`. Missing poses remain NaN
with explicit status codes. This is not an IMU pose or a global absolute pose.

After success, compact reports and initialization selection are retained under
`annotation/vio/<chunk>/<episode>/`, with `left_camera_pose_provenance.json` beside
the feature. The completed episode's `vio` and `vio_zupt` work directories are
removed; failures and raw comparison runs retain diagnostics. Other episodes are
not removed. Existing legacy relative-head-pose files are not migrated automatically.

# L3 头环 VIO

默认读取 headband_left_rgb_rectified / headband_right_rgb_rectified，先执行 L3 双目矫正。
仅使用 observation.headband_imu0.parquet，对应 JSONL 中的 imu0 标定；
不混合 imu1、手腕或眼动 IMU，也不会在 imu0 缺失时自动替换。

按 episode_index 唯一匹配 meta/calibrations.jsonl 的 calibrations.ego_headband，读取：

- camera-camchain：原始双目几何；
- camera_imu-camchain-imucam：T_cam_imu、时间偏移及原相机内参；
- camera_imu-imu：imu0 噪声、采样率等。

按 rostopic 映射左右，校验链式相机顺序，规范化为后端 cam0/cam1。
只在 vio/.../.work/<run-id>/source_calibration 内生成临时 YAML 和来源记录，
不在原数据中伪造 L1 标定文件，不修改原始 JSONL。
继续使用矫正 R1/R2 将相机—IMU 外参转换到矫正相机坐标系。

```bash
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_vio/L3/run.py \
  --source /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001 \
  --openvins-root /share/project/liuyou/humancoin/code/open_vins \
  --workers 2 --zupt-fallback
```

--dry-run 验证输入路径和 JSONL episode 选择，不写文件、不运行后端。
--prepare-only 生成相机配置、双目配对和 IMU CSV，不运行 OpenVINS。
--input-mode raw 支持原始 headband_left_rgb/headband_right_rgb；
尺寸不匹配时需显式 --raw-center-crop-calibration，含义与 L1 相同。

保持各路 Parquet 的共同时间原点，不以 FPS 代替采集时间，不人为叠加时间偏移。
IMU 按 gx/gy/gz（rad/s）、ax/ay/az（m/s²）传给后端；加载器不自动推测或转换单位。
与质检不同，VIO 算法准备阶段仍要求有效且递增的时间戳，并根据真实时间配对左右图像。
质检取消时间检查不等于 VIO 可以接受任意时间序列。

复用 L1 OpenVINS、ZUPT/失败重试、轨迹质量和发布逻辑。
rectified 输出在 vio/ 或 vio_zupt/，并注册 observation.relative_head_pose 和配套状态字段，
更新 info.json、episodes_stats.jsonl；来源相机记录为 L3 左目名称。
raw 输出为 vio_raw/ 或 vio_raw_zupt/，仅保存对照结果，不发布规范位姿。
位姿参考当前 episode 首个有效 IMU，不代表绝对世界轨迹。

CPU 小视频已验证 raw/rectified 配置与输入准备；未在本机运行 OpenVINS 后端。
服务器须编译现有 ov_msckf/build-release/openvins_offline_native。
更多参数、坐标约定及门禁见上级 README.md，算法与 L1 一致。
