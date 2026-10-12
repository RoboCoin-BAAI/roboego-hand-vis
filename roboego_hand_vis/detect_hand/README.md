# 手部采样检测

使用 YOLO 对视频抽样，统计左右手出现比例。此阶段可选，不作为门禁，出现比例不决定是否允许后续处理。

## 检查指定视频

```bash
python -m roboego_hand_vis.detect_hand.run \
  --video /path/to/session/videos/left.mkv
```

默认报告写入 `session/annotation/detect_hand_left.json`，关注 `left_hand_presence_percent` 和 `right_hand_presence_percent`。此命令不回写数据库。

模型和检测参数见 `recommended.yaml`，可使用 `--config`、`--model`、`--device` 等参数覆盖。相对模型路径按 `detect_hand` 模块目录解析。

## 从 L1 数据库批量检测

完成[数据库初始化与连接配置](../database/README.md)，准备好 YOLO 模型后执行：

```bash
python -m roboego_hand_vis.detect_hand.database_run \
  --shared-root ./data/l1 \
  --limit 3
```

`--shared-root` 使用下载时的共享根目录。省略 `--limit` 或设为 `0` 时处理所有当前可领取的数据；可用 `--model /path/to/detector.pt`、`--device cpu` 等参数指定模型和设备。

只领取下载、基础质检、视频粗质检均为 `completed`，且手部检测为 `pending` 的 L1 数据，检测各 session 的 `videos/left.mkv`。运行期间 `detect_hand_status=processing`：

| 结果 | 状态 |
| --- | --- |
| 正常完成，包括手部出现比例低的情况 | `completed` |
| 执行异常 | `failed` |

不根据比例写入 `block`，不自动启动后续阶段。比例仍保存在 JSON 报告中，不新增数据库字段。批次有失败时退出码为 `1`。
