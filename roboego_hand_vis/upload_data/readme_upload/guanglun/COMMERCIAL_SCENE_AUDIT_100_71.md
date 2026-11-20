# 商业场景分类核对：光轮 1–100、简智 1–71

只读核对当前服务器标注，未修改任何分类。与 V2 既有的 11 个商业服务二级分类比较，**本轮没有确认新增的商业二级分类**。此前加入的 `massage_room` 已在基准词表中，不算本轮新增。

后续按用户要求，已将下述 2 条旧 `shelving_area` 统一修正为 `commercial_service / logistics_facility`，仅修改两个分类字段并核对通过。光轮 `logistics_facility` 现为 1155 个，旧标签残留现为 0；以下初次审计表保留修正前的快照。修正报告见 [scene_shelving_correction_report.json](scene_shelving_correction_report.json)。

再次按用户要求，将光轮 5 条、简智 246 条 `commercial_service / massage_room` 全部归入 `professional_work / conference_center`，合计 251 条，已写回并核对仅两个分类字段变化。`massage_room` 在指定范围内已清零，V2 商业服务推荐二级分类由 11 个减为 10 个，全部场景推荐值为 38 个。修正后商业场景样本数为光轮 2613、简智 5775。报告见 [scene_massage_to_conference_report.json](scene_massage_to_conference_report.json)。下表中的 massage_room 数字仅是初次审计时的历史快照。

## 扫描范围

| 数据集 | batch 范围 | 标注数 | 商业场景样本数 | 确认新增类别数 |
|---|---|---:|---:|---:|
| 光轮 guanglun_depth | 1–100 | 9181 | 2618 | 0 |
| 简智 wait_for_publish | 1–71 | 7078 | 6021 | 0 |

合计 16259 份标注。未发现缺失 batch、无标注的空 batch 或 JSON 读取错误；这不等于模型对所有样本都已成功重分类。

## 当前商业场景分布

| area | 中文 | 光轮 | 简智 |
|---|---|---:|---:|
| `retail_store` | 超市 / 零售店 | 953 | 1368 |
| `restaurant` | 餐饮店 | 165 | 87 |
| `beverage_shop` | 饮品店 | 183 | 675 |
| `logistics_facility` | 仓储物流场所 | 1153 | 40 |
| `hotel` | 酒店 | 130 | 2 |
| `hair_salon` | 理发店 | 0 | 1883 |
| `beauty_salon` | 美容店 | 18 | 1718 |
| `entertainment_venue` | 游戏娱乐场所 | 9 | 1 |
| `pet_shop` | 宠物店 | 0 | 1 |
| `performance_venue` | 演出场馆 | 0 | 0 |
| `massage_room` | 按摩理疗服务间（此前已收录） | 5 | 246 |
| `shelving_area` | 货架区（旧标签残留，非 V2 新增） | 2 | 0 |

## 两份旧标签残留

光轮当前出现的唯一词表外商业二级标签是 `shelving_area`，共 2 份：

| batch | 样本 ID | 原任务 |
|---|---|---|
| `batch_0000044` | `9abb1a7b12b4c1cff582236696152cf08b65d830eeccddd7ee02627af84356dc` | 整理口罩包装并放上货架 |
| `batch_0000065` | `b2a299ac9c912f233620d63a6114d7fc63cfe3750556929cbd7f75ae2f6ad443` | 更换货架立柱上的标签 |

两份对应的模型缓存响应均为 `industrial_production / storage_room`。`storage_room` 属于家庭类别，该组合违反已知词表配对，未通过校验，也未生成成功的 `result.json`；原来的商业服务 `shelving_area` 因而保留。不能把这两条算成模型接受的新商业类别。本次未替它们重新选定场景。

## 模型运行记录中的失败

最近保存的两份运行汇总分别记录了：

- `/share/project/liuyou/humancoin/data/guanglun_scene_reports_40_60/summary.json`：3795 个选中样本，3790 个完成、5 个错误；`new_commercial_areas` 为空。
- `/share/project/liuyou/humancoin/data/jianzhi/scene_reclassification_reports/summary.json`：7078 个选中样本，7066 个完成、12 个错误；`new_commercial_areas` 为空。

这些汇总对应各目录最近一次运行的选择范围，不能视为覆盖光轮 1–100 所有历史失败的总账。抽查到的光轮错误包括场景 category/area 配对非法，简智错误包括 HTTP 400。剩余失败需要另行复核；本次仅统计新增类别，未重发模型请求。

完整计数和词表外样本记录见 [commercial_scene_audit_100_71.json](commercial_scene_audit_100_71.json)。
