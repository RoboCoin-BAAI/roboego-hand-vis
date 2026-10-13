# OpenVINS 头部轨迹估计

沿用原 OpenVINS 离线准备与执行流程，不作为质量门禁。

## 处理指定数据

```bash
python -m roboego_hand_vis.vio.run \
  --session /path/to/session \
  --input_mode rectified
```

此命令保持原流程，不回写数据库。

## 从 L1 数据库批量处理

完成[数据库连接配置](../database/README.md)，安装当前 Python 环境的项目 `database` 依赖，并按本阶段 `recommended.yaml` 配置 OpenVINS 二进制和准备脚本使用的 Python 后，在项目根目录执行：

```bash
python -m roboego_hand_vis.vio.database_run \
  --shared-root ./data/l1 \
  --limit 3 \
  --input_mode rectified
```

省略 `--limit` 或设为 `0` 时处理当前所有可领取 session。可用 `--config` 指定其他 VIO YAML。数据库模式固定使用 `rectified`，省略 `--input_mode` 也一样；原手动入口的 raw 模式保留。

只领取下载、头环基础质检、视频粗质检和立体矫正均已 `completed`，且 `vio_status=pending` 的数据。不依赖深度估计、手部估计、深度前门禁或千问标注。

每条数据使用自己的 session，输出固定为 `vio/`：

- `run_manifest.json`：本次输入与准备结果。
- `trajectory/run_report.json`：初始化及位姿数量报告。
- `trajectory/trajectory.jsonl`：轨迹记录。

运行日志为 `annotation/vio_database.log`。时间戳、IMU、标定与视频准备仍使用原算法；起始时间、时长、双目时间匹配容差、IMU 内部/首尾覆盖容差及相机下采样配置仍读取原 YAML。不会自动裁减视频、放宽 IMU 阈值或补充位姿。

领取后写 `processing`。实际 OpenVINS 执行成功，本次重新生成的清单与 session 匹配、报告 `initialized=true`、`pose_count>0`，且轨迹数量与报告一致、记录结构及数值有效后写 `completed`。初始化失败、执行异常、空轨迹或输出不完整写 `failed`，不自动写 `block`。

VIO 初始化前可能没有位姿，因此 `completed` 不表示每个视频帧都有位姿，也不代表轨迹精度已经通过质量评估。仅完成输入准备不会标记成功；数据库入口不提供 `prepare_only`。

多终端原子领取，运行期间不持有数据库事务。失败数据需将其 `vio_status` 从 `failed` 改回 `pending` 后再运行；重试沿用原流程重新执行，已有成功文件不能代替本次结果。状态字段已存在，不需要数据库迁移。
