# 光轮缺失任务名 14 条：原因与补全结果

2026-11-19 已补全服务器这 14 条的 `task_name`，逐条比较备份验证：只增加任务名，其余 JSON 字段一致。没有再次请求模型。标注中的 `objects: []` 保留原状，本轮没有补写物品。

## 为什么缺少

这些原标注均没有 `task_name`，且 `objects` 均为空。原任务名生成没有失败退出，而是将它们列入 review，因此保留了缺失状态。

- 部分建议使用概括名词（exercise equipment、sports equipment、bedding 等），在空物品表和严格文本匹配下未通过。
- 部分描述写 papers、snacks 等复数，旧校验却以完整单数词 paper、snack 做词边界匹配，未识别单复数对应。这是旧校验过严，不是没有任务内容。
- 部分任务含多个目标，或旧的 Subtask 限定不能表达 transfer 等直接操作，模型返回未决。也有模型在空物品表中输出不存在的索引，被拒绝。

本轮读取完整描述、全部子任务及时间，用“一个主要手部操作＋一个主要操作物品”直接分析补全；并非只依据空物品列表判断。

## 写入名称

| batch | 样本 ID 前缀 | task_name | 依据 |
| --- | --- | --- | --- |
| 18 | 90575ee3 | clean chair | 椅子擦拭 54 秒，多于长凳 34 秒；毛巾是工具 |
| 20 | e4e8c3bd | sort shuttlecock | 分类收纳多种体育用品，羽毛球处理占主要时长，盖盒为辅助 |
| 23 | a1d28be8 | stack paper | 将散纸成叠、对齐边缘 |
| 30 | 122b9a6d | transfer food | 筷子是工具，食物从袋转移至盘，主体操作约 73 秒；transfer 在 Atomic 词表中 |
| 37 | d334c055 | organize snack | 反复将零食上架并摆放，货架是目的地 |
| 49 | 77567cf3 | organize paper | 整理纸张 98 秒，放入文件夹为后续 |
| 49 | 843b05ea | organize paper | 整理纸张 33 秒，之后归档 |
| 49 | e01523f2 | fold bed sheet | 三段床单折叠共 180.8 秒，多于毯子 62.9 秒 |
| 50 | fc5e56bd | stack paper | 收集散纸并整理成叠 58.6 秒，开关抽屉为辅助 |
| 64 | 7e9ddd37 | organize item | 反复移动桌面物品 21.2 秒；更换滚筒 11.4 秒、清洁 10.6 秒；多种物品保留统称 item |
| 64 | e57f5585 | organize clothing | 整理悬挂衣物 36.8 秒，多于更换纸 20 秒及清洁衣物 18 秒 |
| 89 | bbcd6561 | tie bag | 收紧袋口并用扎带固定，剪去多余扎带和放入篮子为后续 |
| 96 | 03ea7f86 | sort can | 分类摆放罐、瓶，罐的处理占主要时长，保留 sort 区别于普通整理 |
| 96 | c3500c3b | organize can | 罐摆放 46 秒，多于瓶摆放 31 秒，之后共同调整 |

这些标签只表达主操作和主对象，多对象与辅助步骤仍留在原描述和子任务中。原本复合任务的命名优先级和选择理由记录在 [补全计划](missing_task_names_14_plan.json)。

## 备份与记录

- 服务器报告和备份：`/share/project/liuyou/humancoin/data/task_names_missing_14_20261119/`。
- 原标注：上述目录 `original_annotations/<batch>/<sample>/annotation/`。
- [写入结果](missing_task_names_14_apply_report.json)、[补全前快照](missing_task_names_14_snapshot.json)。

之前全量清单和收敛表是补全前的历史快照，仍记录当时 14 条缺失；本记录是后续修正，不覆盖旧快照。后续正式生成全量写回映射时，应把这 14 条按样本路径补入。
