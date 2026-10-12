# 视频粗质检

采样检查左右目视频的解码、全黑、过曝、低纹理和画面冻结等情况，输出需丢弃或辅助判断的时间区间。

## 检查指定数据

```bash
python -m roboego_hand_vis.video_simple_quality.run \
  --session /path/to/session
```

默认检查 `videos/left.mkv`、`videos/right.mkv`，报告写入 `session/annotation/video_simple_quality.json`。左右目均通过且没有 `drop_before_qwen` 区间时，汇总结果 `passed=true`。此命令不回写数据库。

参数和阈值见 `recommended.yaml`，可通过 `--config` 指定其他配置。原有 `--video` 单视频用法仍可使用。

## 从 L1 数据库批量检查

完成[数据库初始化与连接配置](../database/README.md)后执行：

```bash
python -m roboego_hand_vis.video_simple_quality.database_run \
  --shared-root ./data/l1 \
  --limit 3
```

`--shared-root` 使用下载时的共享根目录；`--limit` 为最多处理条数，省略或设为 `0` 时处理所有当前可领取的数据。数据库模式必须检查左右目。

只领取下载和基础质检均为 `completed`、视频粗质检为 `pending` 的 L1 数据。运行期间标为 `processing`，结果写入 `video_simple_quality_status`：

| 结果 | 状态 |
| --- | --- |
| 门禁通过（`passed=true`） | `completed` |
| 门禁不通过（`passed=false`） | `block` |
| 执行异常或缺少有效门禁结果 | `failed` |

不自动启动后续阶段。批次含 `block` 或 `failed` 时退出码为 `1`。
