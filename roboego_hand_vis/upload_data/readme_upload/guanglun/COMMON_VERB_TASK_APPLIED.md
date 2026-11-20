# common_task_name 已改为动词_task

## 最新：按用户确认的九组规则进一步合并

已应用 [动词合并规则](COMMON_VERB_MERGES.json)：121 种变为 **103 种**。简智 **79 种**、光轮 **76 种**。16259 条全部检查通过，实际更新 817 条（简智 426、光轮 391），错误 0。只改 common_task_name，没有改 task_name 或其他字段，没有创建备份。

两条 throw_task 的描述均明确为把垃圾扔进垃圾桶，因此并入 discard_task。原来的 121 种统计属于下方历史记录。

[最新 103 种分组](common_verb_merged_20261119/common_task_groups.csv)、[最新汇总](common_verb_merged_20261119/summary.json)。服务器报告：`/share/project/liuyou/humancoin/data/common_verb_merged_apply_20261119/`。readme 尚未同步本次新标签。

按用户最新要求，弃用物品大类来组织共用任务名。当前范围内所有结构化标注均已更新：简智 batch 1–71、光轮 batch 1–100，共 16259 个样本、16259 个标注文件。

| 数据 | 标注数／实际修改数 | 共用任务名种数 | 错误 |
| --- | ---: | ---: | ---: |
| 简智 | 7078 | 90 | 0 |
| 光轮 | 9181 | 91 | 0 |
| 合计／去重 | 16259 | 121 | 0 |

规则：从当前细粒度 `task_name` 提取动词，写为 `clean_task`、`organize_task`、`massage_task` 等，不包含物品类别。`origanize_some_object` 这个原细名保留；仅其 common 名称规范为 `organize_task`。不额外合并不同动词。

只修改 common_task_name，逐条读取验证完整 JSON 其他字段一致，task_name、物品、动作、场景、子任务均不变。未创建原标注备份，未请求模型。

- [已实际写入的 121 种分组](common_verb_task_20261119/common_task_groups.csv)
- [结果汇总](common_verb_task_20261119/summary.json)
- [逐条写入及验证报告](common_verb_task_20261119/apply_report.json)
- 服务器报告：`/share/project/liuyou/humancoin/data/common_verb_task_apply_20261119/`。

此前 903 种及物品大类映射为历史方案，已不再用于当前 common_task_name。

readme 的上一轮完整 L1 重建已完成（171 个 batch、16259 份、失败 0），但其 common_task_name 是本次变更前的版本。本轮按“先清洗结构化标注”的要求没有再改 readme，后续需要同步新标签。报告记录每条标注文件字节数变化，可用于同步 readme 的 size_bytes，无需再次扫描视频。
