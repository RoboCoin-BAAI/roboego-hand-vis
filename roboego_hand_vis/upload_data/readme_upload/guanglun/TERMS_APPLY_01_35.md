# 简智与光轮 1–35 物品和动作写回

2026-11-18 已执行。仅替换 `objects[].name` 与 `actions.vocabulary`，保留最新场景、`task_name`、子任务、时间及其他字段。没有新增模型调用。

| 数据 | 样本总数 | 实际修改 | 已与结果一致 | 准备失败，保留原标注 | 写入错误 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 简智 | 3,492 | 2,987 | 470 | 35 | 0 |
| 光轮 | 3,170 | 2,835 | 312 | 23 | 0 |

修正动作校验器将 `return` 误拆为 `re + turn` 的问题，改为只拒绝明确的重复执行变体。复用已有响应恢复 98 个样本（简智 96，光轮 2），没有重发 API 请求。另有一个含 `return` 的样本同时含有无效 `blow_off`，仍保留原标注。

58 条待处理样本主要涉及响应缺失、非法 JSON 或动作短语，不强行改写。逐条路径和实际错误见 [terms_apply_01_35_summary_20261118.json](terms_apply_01_35_summary_20261118.json) 的 `pending` 数组。

## 服务器备份与报告

- 简智：`/share/project/liuyou/humancoin/data/jianzhi/terms_apply_01_35_20261118`
- 光轮：`/share/project/liuyou/humancoin/data/guanglun_terms_apply_01_35_20261118`
- 合并汇总：`/share/project/liuyou/humancoin/data/terms_apply_01_35_summary_20261118.json`

两套目录内均包含：

- `original_annotations/<batch>/<sample>/annotation/episode_*_structured.json`：实际修改文件的原始字节备份。
- `source_snapshot.json`：截取运行中清洗日志得到的 1–35 原始记录。
- `selected_bundle.json`：本次写回采用的独立结果，包含缓存恢复后的记录。
- `cache_recovery.json`：不调用模型的恢复结果。
- `pending_preparation.json`：未写回记录；恢复失败的真实原因另见 `cache_recovery.json` 或合并汇总。
- `apply_progress.jsonl`、`apply_report.json`：写入进度、逐样本状态及统计。

运行中的清洗任务及其原始进度文件没有修改。由于其内存中仍保留旧校验器和历史错误，最终全量清洗文件可能仍显示这 98 条旧失败；当前数据及本次独立报告已包含恢复结果。

写入前检查源标签匹配、备份原文件，并确认替换只影响两个指定字段；写入后逐文件验证实际字节。相关测试共 13 项通过。

执行入口为 [apply_completed_terms.py](apply_completed_terms.py)。它只截取指定批次的完整日志记录，使用独立报告目录；默认不调用模型。可选 `--recover-return-cache` 只复用缓存修复 `return` 误判，不允许发起 API 请求。
