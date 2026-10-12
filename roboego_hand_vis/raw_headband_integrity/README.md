# 头环基础质检

检查原始数据的文件完整性、视频可读性、时间戳对齐与帧数一致性、IMU 信号和时间覆盖，以及音频文件头和音视频时长一致性。

## 检查指定数据

```bash
python -m roboego_hand_vis.raw_headband_integrity.run \
  --session /path/to/session \
  --overwrite
```

报告输出到 `session/annotation/quality_report.json`，`gate.passed` 表示是否通过门禁。此命令不回写数据库。

默认参数和门禁规则见 `recommended.yaml`，可用 `--config` 指定其他配置。默认会修复符合条件的少量重复时间戳；WAV 文件头长度异常时也会直接修复原文件。

## 从 L1 数据库批量检查

完成[数据库初始化与连接配置](../database/README.md)后执行：

```bash
python -m roboego_hand_vis.raw_headband_integrity.database_run \
  --shared-root ./data/l1 \
  --limit 3 \
  --overwrite
```

`--shared-root` 使用下载时的共享根目录；`--limit` 为最多处理条数，省略或设为 `0` 时处理所有当前可领取的数据。

只领取已下载且基础质检为 `pending` 的 L1 数据，运行期间标为 `processing`，结果写入 `raw_headband_integrity_status`：

| 结果 | 状态 |
| --- | --- |
| 门禁通过 | `completed` |
| 门禁不通过 | `block` |
| 执行异常或缺少有效门禁结果 | `failed` |

数据库模式要求启用门禁，不自动启动后续阶段。批次含 `block` 或 `failed` 时退出码为 `1`。
