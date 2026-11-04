# L3 MANUS + VIVE → 世界坐标 21 点

CPU 转换入口：`convert_world_points.py`。默认同时处理左右手，支持单个 session、批次目录和包含多个批次的根目录。不依赖 GPU、视频去畸变、深度或 VIO。

```bash
# 一条数据，生成左右手
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_hand_pose/L3/convert_world_points.py \
  --source /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001/L3-001-2026-10-26-11-07-59

# 整批，最多同时处理 4 个 session
python -m roboego_hand_vis.lerobot_hand_pose.L3.convert_world_points \
  --source /path/to/batch_0000001 --workers 4
```

`--side both|left|right` 默认 both。`--dry-run` 仅列出计划。
重复运行时校验来源、输出文件哈希和变换参数，完全一致则复用。
已有文件无法验证时失败，避免覆盖其他算法结果；`--overwrite` 显式重算本转换器生成的旧结果。
同一 session 不应由两个独立命令同时处理。已接入 L3 CPU pipeline 的 `points` 阶段，顺序为下载 → 质检 → points → 去畸变 → 语义标注。

## 输入与输出

每个 `data/chunk-*/episode_*/` 中读取：

- `observation.hand_skeleton.parquet`：节点 0–24 为左手，25–49 为右手。
- `observation.left_wrist_pose.parquet`、`observation.right_wrist_pose.parquet`：OpenVR 世界中的 Tracker 位姿。

生成同目录下：

- `observation.hand_left_points.parquet`
- `observation.hand_right_points.parquet`
- `hand_left_points_provenance.json`、`hand_right_points_provenance.json`

每个 Parquet 保留骨架原始 `timestamp` 和 `episode_index`，另含 21×3=63 个 float32 坐标；没有模型置信度列，不伪造概率。不改写原始骨架、Tracker 或视频数据。更新 `meta/info.json` 特征声明及 `meta/episodes_stats.jsonl`，统计排除无效采样。

节点顺序与 L1 一致：wrist；thumbCMC/MCP/IP/Tip；食指、中指、无名指、小指的 MCP/PIP/DIP/Tip。
左手选取 `0,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24`；右手全部加 25。

## 固定安装外参

使用当前样本经用户可视化确认的安装假设；采集硬件、安装方式或骨架 SDK 配置不同，需重新确认。

```text
R_tracker_skeleton = [[0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 0]]
t_tracker_wrist = [0, -0.1085, -0.0372] 米

p_world = p_world_tracker
        + R_world_tracker × (R_tracker_skeleton × (p_node - p_root)
                             + t_tracker_wrist)
```

骨架 +X→Tracker +Z，骨架 +Y→Tracker +X，骨架 +Z→Tracker +Y。左右手使用同一旋转；左右手形状差异已经包含在原始骨架内，不额外做反射。骨架 XYZ 按米解释。每帧减去各手根节点后再转换。

Tracker 的 RPY 单位为度，旋转为 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。在骨架时间戳上对 Tracker 平移做线性插值，对姿态做最短路径四元数 SLERP。默认插值区间不得超过 `--max-gap-s 0.03` 秒；超出时间覆盖、源值无效或间隔过大时，整手 21 点为 NaN，不外推或静默填充。

转换器 v2 放宽时间戳要求：骨架保留原始行序及重复时间戳，非有限时间戳对应 NaN 输出；Tracker 丢弃非有限时间戳行，按时间稳定排序，重复时间戳保留原始文件最后一条（整条位置和姿态一起保留）。单个 Tracker 时间点只能用于精确匹配，不外推。处理数量及策略写入 provenance.timestamp_processing。排序不能修复真实时钟回跳造成的跨时间段歧义，倒序数量会记录供复核；原始 Parquet 不修改。没有任何有效同步输出时仍报错。版本更新后，旧 v1 结果需显式 --overwrite 才重算。

所有输出是同一个 **OpenVR TrackingUniverseStanding 世界坐标系**，单位米。场地默认观察视角为 +X 向操作者左侧、+Y 向上、+Z 朝场景深处；观察视角不会改变存储坐标。

所有选中手和 episode 计算成功后再发布结果，失败时不发布半套新输出。来源哈希、安装参数、有效/无效数量写入 provenance，方便复核和复用。
