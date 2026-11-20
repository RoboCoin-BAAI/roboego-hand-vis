# 细粒度 task_name 已写入服务器

按用户最新要求，仅写 `task_name`，不创建标注备份。`common_task_name` 本轮未更新；物品大类尚未用于覆盖细粒度任务名。

| 数据 | 检查并达到目标 | 实际修改 | 无需修改 | 错误 |
| --- | ---: | ---: | ---: | ---: |
| 简智 1–71 | 7078 | 2822 | 4256 | 0 |
| 光轮 1–100 | 9181 | 2759 | 6422 | 0 |
| 合计 | 16259 | 5581 | 10678 | 0 |

来源：`remaining_task_review_20261119/all_sample_preview.csv`。578 条共享标签按用户指定写为 `origanize_some_object`；之前已补的 14 条包含在检查范围中。

每条写入前核对当前任务名，原子写入后核对完整 JSON 与写入前相比仅 `task_name` 不同；未更改文件也读取验证任务名已符合映射。没有复制原标注备份，报告仅记录路径、目标名、修改状态、验证结果与结果文件哈希。

- [本地结果汇总](task_names_fine_apply_no_backup_20261119_summary.json)
- 服务器报告：`/share/project/liuyou/humancoin/data/task_names_fine_apply_no_backup_20261119/apply_report.json`
- 脚本：`bulk_apply_task_names.py`
