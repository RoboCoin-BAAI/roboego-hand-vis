# 主动作＋主物品：收敛映射建议

使用全量 2026-11-19 快照、词法映射、已有模型缓存以及明确的短语规则，生成一版可审阅的外部收敛表。本轮未请求模型、未修改任何标注；所有语义归并都是建议。

## 范围和结果

| 数据集 | 原不同名称 | 已给出统一名称的旧名 | 对应统一名称数 | 覆盖样本 | 尚待复核样本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 简智 1–71 | 1777 | 1090 | 764 | 6359 | 719 |
| 光轮 1–100 | 1674 | 1418 | 1234 | 8828 | 339 |
| 跨数据集去重 | 3213 | 2271 | 1785 | 15187 | 1058 |

总样本 16259；另有光轮 14 条缺少任务名。候选映射覆盖 93.4% 的样本。1785 不是全量最终类别数，剩余 942 个不同长尾名称仍保留旧名待复核；如果把这些旧名也算入，预览中的全量不同名称为 2726 个（简智 1450、光轮 1490）。

4524 个样本的任务名在候选中发生变化，其余已覆盖样本保留原名。覆盖率不等于准确率，不代表已审核通过。

## 可直接查看的表

- [任务分组 groups.csv](hand_task_consolidation_20261119/groups.csv)：1785 个候选标准名称、各自关联的旧名及两套数据样本数。
- [全部映射 mapping.csv](hand_task_consolidation_20261119/mapping.csv)：3213 个旧名，含结果、依据来源、规则和待复核原因。
- [逐样本预览 sample_preview.csv](hand_task_consolidation_20261119/sample_preview.csv)：16259 行，按路径追踪旧名和候选；未映射项 proposed 为空，不能当成空值覆盖。
- [剩余长尾 remaining_review.csv](hand_task_consolidation_20261119/remaining_review.csv)：942 个不同名称，涉及 1058 样本。
- [带校验提示的候选 caution_candidates.csv](hand_task_consolidation_20261119/caution_candidates.csv)：30 个名称。旧模型填的证据/原词字段不一致，但目标物品确实在旧名中，保留为有提示的候选，不自动认可。
- [完整映射](hand_task_consolidation_20261119/mapping.json)、[汇总](hand_task_consolidation_20261119/summary.json)、[规则清单](hand_task_consolidation_20261119/rules.json)。

## 收敛规则

优先保留已有、可解析的短名称，不因为模型曾选出另一动作就重写它。例如 `organize cable` 不自动改成 `coil cable`。复杂名称才使用通过校验的模型候选或明确短语规则，且仍是建议。

1. 沿用显式单复数和词法同义规则，保留多词物品概念与有意义的子类。`cosmetic → cosmetics`、`toiletry → toiletries`、泛称 `object → item`、`skincare product → skin care product` 作为本轮新增建议；不把 `item` 与具体商品类别合并。
2. `wipe → clean`；普通物品摆放、收纳白名单中的 `arrange/tidy → organize`。头发、假发、花束、展示造型等不套用普通收纳规则。
3. 提议将物品表面 `scrub → clean`，排除身体护理目标；将 `sweep/mop/vacuum floor/ground/carpet/rug → clean`。这些是任务标签层级的方式归并，并非动作完全同义；具体方式保留在原有动作词表和子任务中。需要审核后才能用于写回。
4. `sort/stack/organize`、`pack/seal/wrap`、`cut/trim/shave`、`wash/clean/polish` 等继续分开。
5. `perform/give facial massage → massage face`；明确的身体部位按摩短语统一，去掉 customer 等受服务者前缀；`massage back and head` 这类多部位且主对象不明确的短语不靠此规则强行选一个。
6. `inflate basketball with air pump → inflate basketball`；`file customer fingernails → file nail`。只处理明确模式，不把所有 with 后面的内容盲目删除。
7. 不生成以 bathroom、room、workspace 等场景/抽象目标为主物品的标签，不把目的地代替主要操作对象。

## 实际归并示例

| 候选统一名称 | 关联旧名称种数 | 覆盖样本 |
| --- | ---: | ---: |
| roll hair | 42 | 53 |
| massage back | 24 | 72 |
| massage face | 22 | 102 |
| massage abdomen | 19 | 40 |
| perm hair | 14 | 21 |
| clean floor | 8 | 160 |

以上分组来自词法、明确短语或已有模型候选，仍需检查具体来源。不能仅凭新名相同就忽略病例/操作流程差异。

## 剩余项为什么保留

多个同等重要动作/物品、泛化服务名称没有具体手部动作、描述冲突、模型响应损坏且旧名无法可靠简化，均保留待复核。优先审查 `groups.csv` 中成员最多的分组和带 caution 的候选，比逐条检查全部样本更有效。

脚本：`consolidate_hand_task_mapping.py`。仅生成外部表，不提供标注写回入口。针对性的 11 项测试通过，覆盖任务边界、身体/物品清洁区分、明确短语和模型校验。
