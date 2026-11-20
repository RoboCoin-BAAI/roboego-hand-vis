# 第二轮动词收敛已写入

按用户确认的力度合并 common_task_name，不应用低频 other_task 规则。只修改该字段，task_name 及其他 JSON 字段逐条验证不变，不创建原标注备份。

| 数据 | 总标注 | 本轮修改 | 最新共用任务种数 | 错误 |
| --- | ---: | ---: | ---: | ---: |
| 简智 1–71 | 7078 | 501 | 64 | 0 |
| 光轮 1–100 | 9181 | 1717 | 61 | 0 |
| 合计／去重 | 16259 | 2218 | 87 | 0 |

上一轮 103 种，本轮为 87 种。仍有条件保留的动词，因此不是将所有来源词机械删除。

新增合并：放置/收纳/补货/堆叠/对齐/分类归 organize；普通物品收集、悬挂收纳也归 organize。包裹、封口归 pack；安装和装配插接归 assemble；粘贴与附着归 attach；捆绑、拧紧和固定归 secure；修剪归 cut。第一轮规则继续有效，已展开为直接目标，避免 bind→tie→secure 等链式遗漏。

上下文例外：收集垃圾归 clean；明确晾干的悬挂归 dry；安装吊挂物保留 hang；明确插电充电的 insert/plug/connect 保留原插接动词；有 unscrew/loosen 描述时不将 screw 强行当固定。判断依据为当前任务名和描述，未请求模型。

- [最新 87 种任务分组](common_verb_round2_20261119/common_task_groups.csv)
- [结果汇总](common_verb_round2_20261119/summary.json)
- [完整结果](common_verb_round2_20261119/apply_report.json)
- [平铺动词规则](COMMON_VERB_MERGES_ROUND2.json)，上下文例外在 `clean_common_verb_task.py --round2-context` 中。
- 服务器报告：`/share/project/liuyou/humancoin/data/common_verb_round2_apply_20261119/`。

没有写入 other_task，低频类别尚未删除。readme 本轮未同步，仍需在最终标签确定后更新。
