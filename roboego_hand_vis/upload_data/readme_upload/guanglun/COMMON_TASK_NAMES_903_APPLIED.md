# common_task_name 903 种映射已写入

2026-11-19 按用户要求写入服务器，仅更新 `common_task_name`，沿用不创建标注备份的要求。

| 数据集 | 写入并验证样本 | common_task_name 种数 | 错误 |
| --- | ---: | ---: | ---: |
| 简智 1–71 | 7078 | 401 | 0 |
| 光轮 1–100 | 9181 | 680 | 0 |
| 合计／合并去重 | 16259 | 903 | 0 |

全部 16259 条发生更新。每条先核对 `task_name` 与已批准映射一致，原子写入后比较完整 JSON，确认只有 `common_task_name` 改变。`task_name`、物品、动作、场景、子任务及其他字段保持原值。

写入依据：[逐样本映射](common_tasks_31_20261119/sample_mapping.csv)。细粒度任务名仍为简智 838 种、光轮 1272 种，合并去重 1888 种。

- [结果汇总](common_task_names_903_applied_summary.json)
- 服务器报告目录：`/share/project/liuyou/humancoin/data/common_task_names_903_apply_no_backup_20261119/`
- 逐条结果：该目录的 `apply_report.json`、`progress.jsonl`，含路径、目标值、验证状态和结果文件哈希。
- 脚本：`bulk_apply_common_task_names.py`。

本次没有创建原标注副本；报告不等同于原文件备份。
