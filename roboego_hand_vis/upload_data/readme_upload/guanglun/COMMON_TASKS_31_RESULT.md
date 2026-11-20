# 31 类物品大类与 common_task_name 预览

**本方案已停用。** 用户随后要求 common_task_name 只用“动词_task”，现已全部写入，合并去重 121 种。详见 [最新结果](COMMON_VERB_TASK_APPLIED.md)。下文均为历史记录。

**最新状态：903 种映射已正式写回服务器。** 简智 7078 条、光轮 9181 条全部更新并验证，错误 0；只改 common_task_name，不备份。见 [写入结果](COMMON_TASK_NAMES_903_APPLIED.md)。下文“预览／未写回”描述的是此前整理阶段。

## 最新更新：泛称统一为 item

在已知具体类别优先的前提下，将剩余 product/object/item/goods/supply/material 等泛称及形状、位置限定的泛称统一为 item。例如 `cylindrical object`、`long item`、`household item`、`office supply` 不再单立共用类别。`skin care product` 仍为 cosmetics，`medicinal material` 仍为 medicine，`paper material` 仍为 paper_material。

578 条指定兜底的细粒度 `task_name` 仍原样保留 `origanize_some_object`；本次仅在 common 预览中统一为 `organize_item`。不同操作动词仍区分，不把 sort/stack/pack 等合并成 organize。

最新 common 名称预览：**简智 401 种、光轮 680 种、合并去重 903 种**，较上一版 950 种减少 47 种。31 主类覆盖 15039 条，明确独立类别 442 条，其他保留细类 778 条，合计 16259 条。细粒度 task_name 的逐样本值经比较完全不变。

这一版仍是外部预览，没有改写服务器 common_task_name。下方 950 等数字为此前版本记录。

## 最新更新：按用户确认补齐规则并保留独立类别

- 鞋带 37 条归衣物，吸管 22 条归餐具，球 17 条归运动用品，香肠 5 条归食品。
- 地面 167 条、水槽 118 条、镜子 47 条、门 23 条等保留独立类别，分别使用 `floor`、`sink`、`mirror`、`door`，不并入家具。
- 对原来未定的 mask/glass/liquid 共 59 条，额外读取场景和完整子任务。面膜归化妆品、杯具归餐具、玻璃柜台归家具、洗手液归日常洗护、甲油胶归指甲油；描述仍不明确时保留 mask/glass/liquid 细类，不猜用途。
- 其他不适合 31 主类的物品同样保留具体类型，标记为 `retained_fine`，不再当成空缺待处理项，也不全部压成 item。

| 当前处理方式 | 样本数 |
| --- | ---: |
| 归入 31 主类 | 14973 |
| 明确保留的建筑设施等独立类别，以及未明 mask/glass/liquid | 442 |
| 其他保留原具体类型 | 844 |
| 合计 | 16259 |

最新 common 名称预览：简智 **407** 种、光轮 **722** 种、合并去重 **950** 种。这包含保留的独立类别，不是只有 31 个物品枚举的组合。所有样本均有非空类别与 common 名称；细粒度 task_name 不变，common_task_name 仍未写回标注。

[更新后的分组](common_tasks_31_20261119/common_task_groups.csv)、[逐样本映射](common_tasks_31_20261119/sample_mapping.csv)、[独立类别统计](common_tasks_31_20261119/retained_category_counts.csv)。`unmapped_nouns.csv` 现在列的是未并入主类、已按细类保留的词；`has_unmapped_noun` 表示组中含保留细类，不表示缺少结果。

以下为更新前的统计记录，1399 条类别未定等数字不代表当前处理状态。

已按确认的 31 类建立定义、明确词表及上下文规则。使用已写回的细粒度任务名生成预览，不修改 `task_name`，本轮也没有覆盖服务器 `common_task_name`。

## 结果

| 数据 | 总样本 | 已归入 31 类 | 类别未定 | 细粒度任务名 | common 名称预览种数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 简智 | 7078 | 6730 | 348 | 838 | 412 |
| 光轮 | 9181 | 8130 | 1051 | 1272 | 742 |
| 合并去重 | 16259 | 14860 | 1399 | 1888 | 972 |

已归类覆盖约 91.4%；这部分合并后共 502 个 common 名称。全量预览 972 种还包含未归类条目保留的细名，不能解释为已全部完成大类映射。

31 类并非完整覆盖所有现实物品。例如地面、墙壁、门窗、水槽等建筑表面/设施和生活植物没有合适类别；还有 rod、mask、glass、brush 等需要上下文明确。当前不能确定的 1399 条保留细名作为 common 预览，不以 item 隐藏目录缺口。`unmapped_nouns.csv` 按词及规则去重列出，并附两套数据的样本数。

## 文件

- [31 类定义及词表](OBJECT_CATEGORIES_31.json)：英文枚举、中文名称、定义、成员和判定顺序。
- [各大类数量](common_tasks_31_20261119/category_counts.csv)：31 行中英对照统计。
- [全量任务分组](common_tasks_31_20261119/common_task_groups.csv)：common 名称、两套数据数量、细粒度任务成员；`has_unmapped_noun` 标出尚未归类的组。
- [逐样本映射](common_tasks_31_20261119/sample_mapping.csv)：保留 task_name，另列主物品、类别、common 名称和依据。
- [类别未定词表](common_tasks_31_20261119/unmapped_nouns.csv)。

## 明确规则

| task_name 保留 | common_task_name 预览 | 大类 |
| --- | --- | --- |
| clean chair | clean_furniture | 家具 |
| clean table | clean_furniture | 家具 |
| wash cup | wash_tableware | 餐具 |
| wash bowl | wash_tableware | 餐具 |
| apply cream | apply_cosmetics | 化妆品及护肤品 |
| apply hair dye | apply_hair_dye | 染发剂 |
| apply nail polish | apply_nail_polish | 指甲油 |
| massage face | massage_body | 身体 |
| massage back | massage_body | 身体 |
| organize book | organize_paper_material | 纸质资料 |
| organize pen | organize_stationery | 文具 |
| clean bottle | clean_container | 容器 |
| clean surface | clean_item | 未细分物品 |

先判断被操作的概念，再分类：擦乳液瓶归容器，涂乳液归化妆品；餐杯优先餐具，不因能盛东西而归一般容器；nail drill 为美容工具，不被 drill 机械归通用工具。染发剂与指甲油独立保留。普通毛发操作归 hair，按摩不论部位统一 body。

上下文仅用于明确同一物品词的含义，不重新选择主动作或主对象。多个语义均可能时保留未定，不强行制造确定性。包含用户指定的 578 条兜底标签 `origanize_some_object`，其类别记录为 item、common 名称保持原样，不自动修正拼写。

验证：31 类数量、材料与容器区分、部位按摩、专用工具优先、上下文歧义与目录缺口的针对性测试通过；全量 16259 行，路径唯一，common 字符串满足现有 schema 的下划线格式。
