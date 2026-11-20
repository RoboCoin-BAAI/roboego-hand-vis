# 全量任务名称清单与分阶段映射规则 V1

2026-11-19 从服务器直接读取简智 batch 1–71、光轮 batch 1–100。只收集和生成映射预览，没有修改标注，没有调用模型。

## 当前统计

| 数据集 | 样本总数 | 缺任务名 | 原始任务名种数 | 格式、物品规范后 | 再归并动词后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 简智 | 7078 | 0 | 1777 | 1712 | 1686 |
| 光轮 | 9181 | 14 | 1674 | 1654 | 1619 |
| 跨数据集合并去重 | 16259 | 14 | 3213 | 3132 | 3079 |

读取错误 0。数字来自本次采集快照，不代表所有标注清洗阶段均已完成。中间阶段同时包含大小写和空白规范，不能把全部降幅都归因于物品单复数。

3359 个样本的名称在预览中发生变化。保守规则不能让类别骤减：跨数据集有 1276 个不同的复杂或未识别名称，保留原文字内容，仅允许格式规范，列入复核；不得把它们解释为已经完成语义归并。`ready` 仅表示可按词法规则解析，不代表任务语义经过审核。

## 文件

- [逐样本任务名清单](task_name_mapping_20261119/all_sample_task_names.csv)：16259 条，含数据集、路径、当前任务名和任务描述。
- [完整映射表](task_name_mapping_20261119/task_name_mapping.csv)：3213 个原始名称，保留原名、物品规范后名称、动词归并后名称、规则和两套数据计数。
- [物品映射表](task_name_mapping_20261119/object_mapping.csv)：可解析任务的完整主物品短语、规范词及计数，不截断短语。
- [复杂任务复核表](task_name_mapping_20261119/review_tasks.csv)：不强制简化的名称。
- [缺失任务名](task_name_mapping_20261119/missing_task_names.csv)：光轮 14 条。
- [机器可读规则](task_name_normalization_rules.json)、[含样本依据的映射](task_name_mapping_20261119/mapping.json)、[统计](task_name_mapping_20261119/summary.json)。

这些物品规则只用于 `task_name` 中的物品部分，不再次清洗或改写 `objects[].name`。

## 顺序一：格式和物品

1. 统一小写，去掉首尾空白，连续空白合并。
2. 完整物品短语先匹配明确同义/拼写映射，如 `clothes → clothing`、`eyeglasses/spectacles → glasses`、`hairdryer/hair drier → hair dryer`。
3. 按显式复数词头表规范最后一个单词，如 `boxes → box`、`shelves → shelf`、`nail tips → nail tip`。保留前面的物品修饰和细分类别。
4. 不使用直接去掉 s/es 的算法。`scissors`、`glasses`、`pants`、`cosmetics`、`toiletries`、`glass`、`lip gloss` 等保留。
5. 保留物品类别差异：`box ≠ carton`、`counter ≠ table`、`cloth ≠ towel`、`bottle ≠ jar`、`hair ≠ wig`、`item ≠ product`。
6. 不删除 `customer`、`mannequin` 等原有限定语；是否进一步归并需另定规则，不能在单复数清洗中悄悄删除。

## 顺序二：动词

### 纳入当前预览

| 映射 | 定义与范围 | 排除或限制 | 正例 | 反例 |
| --- | --- | --- | --- | --- |
| `wipe → clean` | 按用户确认，将擦拭这一操作方式提升为任务名中的清洁目的；物品部分保留 | 只改变任务名称，不修改动作词表/子任务；多目标长句不自动替换 | `wipe boxes → clean box` | `wipe counter and organize bottles` 保留长句待复核 |
| `organise → organize` | 英美拼写统一 | 不扩展到含义不同的动作 | `organise box → organize box` | `sort box` 保留 |
| `colour → color` | 英美拼写统一 | 不把染色与清洁等其他任务合并 | `colour hair → color hair` | `wash hair` 保留 |

`wipe → clean` 是任务层级的目的归并，不宣称它们在所有语境中完全同义。原始动作方式仍可在现有子任务中追溯。

### 仅列候选，不计入当前归并结果

| 候选 | 可归并条件 | 必须排除的情况 |
| --- | --- | --- |
| `arrange/tidy → organize` | 都是在表达物品有序摆放或收纳 | 艺术摆设、头发造型、装饰等，不能只凭动词替换 |
| `scrub → clean` | 清除对象表面污物 | 磨削加工、皮肤去角质等 |
| `wash/rinse → clean` | 明确决定不区分清洁方式 | 染发后的冲洗、去除处理药剂等不能等同一般清洁 |
| `refill → fill` | 统计明确不区分首次装填与补充 | 如果补充过程本身是研究目标，则保留 |
| `check → inspect` | 核验或检查对象 | 描述不明确的 check 不能直接接受 |

候选目标记录在映射表 `candidate_merges` 列，仅供审阅；`proposed` 列没有应用它们。

### 明确保留

- `organize / sort / stack`：整理、分类、堆叠目标不同。
- `pack / seal / wrap`：装包、封口、包裹不同。
- `cut / trim / shave`：不能无条件合并普通剪切、修整和剃除。
- `apply / fill / pour`：施加、装填、倾倒不同。
- `clean / polish`：去污与抛光不同。
- `assemble / install / repair`：组装、安装、维修不同。

## 完整示例

| 原名称 | 物品规范后 | 动词归并后 |
| --- | --- | --- |
| `Wipe Boxes` | `wipe box` | `clean box` |
| `organize clothes` | `organize clothing` | `organize clothing` |
| `wipe counter` | `wipe counter` | `clean counter` |
| `wipe mirror` | `wipe mirror` | `clean mirror` |
| `stack box` | `stack box` | `stack box` |
| `wipe counter and mirror` | 保留，待复核 | 保留，待复核，不升级成 bathroom |

旧名到新名的映射全部保留。多动作、多个对象、介词关系、不认识的动词和任务缺失都不自动猜测或删减。本轮没有再次根据视频目标重分类。

## 复现

```bash
python build_task_name_mapping.py \
  --inventory task_names_inventory_20261119.json \
  --output task_name_mapping_new
```

输出目录须为空。脚本没有标注写回入口。映射规则及例外经过针对性测试：单复数与动词处理顺序、名词例外、多动作保护、不同任务目标保留。
