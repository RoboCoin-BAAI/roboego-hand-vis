# 光轮前 39 个 batch 场景清洗记录

首次清洗于 2026-11-18 写回服务器并完成逐文件核对：3541 个样本中，2533 份发生修改，1008 份原值已一致，错误 0。首次报告见 [scene_apply_report_01_39.json](scene_apply_report_01_39.json)。

随后按用户提供的场所经验修正 73 份标注，全部写回并核对通过；报告见 [scene_room_corrections_report.json](scene_room_corrections_report.json)。再将 16 个运动器材整理收纳区样本归入 `indoor_sports_venue`，已写回核对，报告见 [scene_sports_corrections_report.json](scene_sports_corrections_report.json)。

最新按用户要求以家庭场景为前提，复分原 950 个 other 样本，其中 916 个归入家庭的已有二级分类，保留 34 个 other。详细结果见 [HOUSEHOLD_OTHER_REVIEW.md](HOUSEHOLD_OTHER_REVIEW.md)，核对报告见 [scene_household_corrections_report.json](scene_household_corrections_report.json)。这次房间归属是基于家庭前提及任务文本的推断，没有查看视频核实房间。

各次均仅修改 `scene_classification.category`、`scene_classification.area` 的值，其他字节保持不变，未新增字段。以下统计及总清单为修正后的最终结果。

再按用户要求，将 2 个 `badminton_court` 样本归入 `indoor_sports_venue`，已写回并核对通过；室内体育馆现共 97 个样本。报告见 [scene_badminton_corrections_report.json](scene_badminton_corrections_report.json)。随后将 6 个 `meeting_room` 样本统一归入 `conference_center`，已写回核对，报告见 [scene_meeting_corrections_report.json](scene_meeting_corrections_report.json)。之后按用户指定，将 `massage_room` 统一归入 `professional_work / conference_center`，当前没有额外新增二级类别。

分类来源：`D:/data/annotation_statistics/理想/HumanCOIN_Sence.xlsx`。一级分类固定，二级分类允许新增。

首次判断只使用 `task_description` 与 `scene_classification.original_description`，不使用视频，也不以旧 category/area 决定新标签。按相同场景描述及任务语义分组分析，形成保守的可复现分类草稿，再复核并更正易混淆分组；场所依据不足时归入 `other`。用户后续提供的场所经验优先于文本推断。

## 用户修正

| 原二级分类 | 最终二级分类 | 样本数 |
|---|---|---:|
| `dressing_area` | `bedroom`（卧室） | 25 |
| `home_fitness_area` | `living_room`（客厅） | 25 |
| `home_recreation_area` | `living_room`（客厅） | 8 |
| `pet_care_area` | `living_room`（客厅） | 3 |
| `shoe_care_area` | `living_room`（客厅） | 8 |
| `food_service_area` | `restaurant`（餐饮店） | 4 |
| `sports_equipment_area` | `indoor_sports_venue`（室内体育馆） | 16 |
| `badminton_court` | `indoor_sports_venue`（室内体育馆） | 2 |
| `meeting_room` | `conference_center`（会议中心） | 6 |
| `massage_room` | `conference_center`（会议中心，用户指定归并） | 1 |

本轮用户明确场所的修正后，卧室共 166 个样本、客厅共 70 个样本、餐饮店共 4 个样本、室内体育馆共 95 个样本，other 为 950 个。上述七个临时分类已从最终分类及新增分类统计中移除。之后对 other 的家庭复分继续增加了各家庭房间样本数，以最新总清单为准。

## 一级分类统计

| 一级分类 | category | 样本数 |
|---|---|---:|
| 家庭 | `household` | 1915 |
| 工业生产 | `industrial_production` | 49 |
| 商业服务 | `commercial_service` | 904 |
| 专业工作 | `professional_work` | 542 |
| 公共场所 | `public_space` | 97 |
| 其他 | `other` | 34 |

公共场所的英文一级标识使用 `public_space`，涵盖室内和户外公共活动场所。

## 本次新增二级分类

| 一级分类 | area | 中文说明 | 样本数 |
|---|---|---|---:|

此前临时分类均已归并，本次新增二级分类数量为 0。

## 清单及工具

- [scene_classification_manifest.json](scene_classification_manifest.json)：全部样本的目标分类及文本依据。
- [scene_other_samples.csv](scene_other_samples.csv) / [scene_other_samples.json](scene_other_samples.json)：待后续处理的 34 个 other 样本。
- [scene_household_review.csv](scene_household_review.csv)：原 950 个 other 样本的家庭房间复分结果，保留原任务组号。
- [scene_taxonomy_and_counts.json](scene_taxonomy_and_counts.json)：Excel 原始 38 个二级枚举、六个一级分类映射、新增分类和统计。
- `prepare_scene_review.py` / `finalize_scene_review.py`：分组分类草稿及复核修正规则。
- `apply_scene_classification.py`：预检及写回工具，只替换原 JSON 两个已有字段的值，保留其他字节。

服务器清洗记录目录：`/share/project/liuyou/humancoin/data/guanglun_scene_cleaning_20261118/`。原文件副本位于该目录 `original_annotations/` 下，不在上传数据目录中。

本次不修改 Argus 的旧 V1 词表和 Schema；新增二级分类及 `public_space` 应按本次新词表解释，旧 V1 校验规则不适用于这些新分类。
