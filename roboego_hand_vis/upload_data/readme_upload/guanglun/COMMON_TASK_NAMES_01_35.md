# 任务大类 common_task_name

保留细粒度 `task_name`，新增顶层 `common_task_name` 用于更大范围的任务归并。范围为简智和光轮 batch 1–35，共 6,662 条，包括细粒度任务名待复核、物品动作待处理及没有 task_name 的样本。

```json
{"task_name": "wipe table", "common_task_name": "object_cleaning"}
```

分类依据是任务描述及子任务的整体目的，不按动作词频或场景单独判定。多物品同一目的不再被迫选择一个物品，无明确主目的时使用 `other`。

## 2026-11-18 执行结果

| 数据 | 已写入样本 | 实际大类数（含 other） | other 待复核 | 写入错误 |
|---|---:|---:|---:|---:|
| 简智 1–35 | 3,492 | 20 | 86 | 0 |
| 光轮 1–35 | 3,170 | 15 | 67 | 0 |
| 合计 | 6,662 | 21（两套并集） | 153 | 0 |

全部样本已新增 `common_task_name`，原 `task_name` 已逐样本核对保持不变。此层级不再要求必须先成功规范细粒度任务名，因此覆盖此前 58 条物品/动作待处理样本及 4 条缺少任务名的样本。没有改动这些样本原来的物品/动作或任务名。

- [大类数量统计 CSV](common_task_counts_01_35_20261118.csv)
- [完整分组索引 CSV](common_task_index_01_35_20261118.csv)：6,662 条，包含数据集、场景、原任务名、大类和标注路径。
- [场景与大类分组统计 CSV](common_task_groups_01_35_20261118.csv)：171 个分组，含 other 分组。
- [other 样本及理由 JSON](common_task_other_samples_01_35_20261118.json)
- [执行汇总 JSON](common_task_summary_01_35_20261118.json)

后续按 `(category, area, common_task_name)` 合并时，先排除 `common_task_name=other`，这些样本单独复核，不当作同一种实际任务合并。

## 分类表

| 英文枚举 | 中文大类 |
|---|---|
| `object_cleaning` | 物品清洁 |
| `environment_cleaning` | 环境清洁 |
| `object_organizing` | 物品整理收纳 |
| `object_packaging` | 物品包装拆包 |
| `object_labeling` | 物品标识贴标 |
| `object_transport` | 物品搬运装卸 |
| `object_assembly` | 物品装配拆装 |
| `object_maintenance` | 物品维护保养 |
| `object_inspection` | 物品检查核验 |
| `object_processing` | 物品加工制作 |
| `object_filling` | 物品分装补充 |
| `object_operation` | 设备操作设置 |
| `food_preparation` | 食品饮品制作分装 |
| `hair_styling` | 美发造型 |
| `hair_care` | 头发清洁护理 |
| `nail_care` | 美甲及指甲护理 |
| `skin_care` | 皮肤清洁护理 |
| `makeup_application` | 化妆修饰 |
| `body_care` | 身体按摩理疗 |
| `writing_drawing` | 书写绘画 |
| `sports_recreation` | 运动游戏 |
| `other` | 其他待复核 |

## 主要边界

- 清洗杯具、桌面、衣物和工具属于 `object_cleaning`；清扫地面、垃圾和整体房间属于 `environment_cleaning`。
- 收纳、排序、叠放、折衣和卷线属于 `object_organizing`；为发货装袋、装箱、封口属于 `object_packaging`。
- 美发店整理工具仍属于 `object_organizing`，不能仅因位于美发店就归到 `hair_styling`。
- 剪发、烫发、编发、拉直、染发归 `hair_styling`；以清洗、护理或单纯吹干头发为目的归 `hair_care`。
- 护肤、装饰化妆、按摩理疗分别归 `skin_care`、`makeup_application`、`body_care`。
- 厨房洗碗、擦台面仍属 `object_cleaning`；烹饪、备菜和制作饮品属 `food_preparation`。

## 执行与备份

脚本：[classify_common_tasks_qwen.py](classify_common_tasks_qwen.py)，使用 `qwen3.7-plus` 分组判断。原始模型响应和输入均单独缓存，小样结果会在全量运行中复用。

服务器报告目录：`/share/project/liuyou/humancoin/data/common_tasks_01_35_20261118`。

仅写 `common_task_name`，保留任务名、场景、物品、动作和时间等其他字段。原标注备份到报告目录的 `original_annotations/<dataset>/<batch>/<sample>/annotation/`。已有不同的 `common_task_name` 不会被直接覆盖。

Argus V1/V2 的本地 schema 已允许该可选字段；服务器现有 V1 schema 同步增加可选定义，原 schema 备份在报告目录的 `schema_backups`。旧字段 `common_task` 保留。
