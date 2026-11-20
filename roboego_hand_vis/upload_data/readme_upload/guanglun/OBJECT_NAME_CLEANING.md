# 光轮与简智物体名称清洗

入口：[clean_object_names_qwen.py](clean_object_names_qwen.py)。Python 3.10+，默认模型 `qwen3.7-plus`，复用 Argus 的 `review/qwen_endpoint.py` 网关和密钥配置。

## 本次字段范围

**只修改 `objects[].name`**。不修改 `attributes`、`location`、`original_attributes`、子任务的 `object` / `destination`、动作原文、时间、场景或其他字段。即使归一化后出现重复名称，也保留原来的对象数量和顺序，不删除、合并或新增物体。

分类依据是原物体名称，任务描述仅用于语义消歧，不从任务文本补充新的物体。本脚本不读视频、不抽帧，仅调用文本模型处理名称。

## 名称规则

- 参考 `D:/data/annotation_statistics/理想/HumanCOIN_Object.xlsx`，优先对齐其中简洁英文名；不是封闭枚举，表外但含义清楚的名词可以保留并列入 `new_names.json`。
- 去掉冠词、数量、量词、颜色、大小、形状、材质、状态、方位、品牌和型号等修饰。
- 自然情况下优先使用单数；`scissors`、`glasses`、`pants` 等类别本身为复数形式时保留。
- 保留构成完整物体类别的复合名词，例如 `washing machine`、`remote control`、`first aid kit`、`cotton swab`。不机械地只保留最后一个词。
- 原数据里已经存在的人物、角色、身体部位仅规范名称，不删除，也不改成其他物体。
- 无法确定的名称保留原值，在外部 `unresolved_names.csv` 中等待复核。个别候选名称仍含修饰或缺少解释时，保留该物体原名并列入待复核，其他合法项继续处理；模型输出索引、数量或响应结构不合法时，整份标注不写回，记录错误。

| 原名称 | 归一化名称 |
|---|---|
| `two red plastic cups` | `cup` |
| `large cardboard boxes` | `box` |
| `a pair of scissors` | `scissors` |
| `strands of hair` | `hair` |
| `washing machine` | `washing machine` |

## 参考词表

[object_vocabulary.json](object_vocabulary.json) 从 Excel 中有数字序号的行导出，共 636 个名称。Excel 表名/尾注虽提到 639，但实际可读取的编号词条为 636 条；尾部“639 个标准物品词”等文字已排除。不会修改原 Excel，也不会将其当成严格白名单；其中带修饰或别名的条目仍按上述规则判断。

重新导出：

```bash
python export_object_vocabulary.py \
  --source /path/to/HumanCOIN_Object.xlsx \
  --output object_vocabulary.json
```

## 服务器运行

工具已同步到 `/share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools/`。与场景脚本共享 `apply_scene_classification.py`、`reclassify_scenes_qwen.py`；复制使用时也需要保留这两个依赖文件，以及 `object_vocabulary.json`。不需要 FFmpeg。

沿用已配置的 `XMSXB_API_KEY`（可选 `XMSXB_BASE_URL`），或 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`。先使用项目 Python 环境：

```bash
export PATH=/share/project/liuyou/humancoin/envs/ego/bin:$PATH
cd /share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools
```

光轮前 39 个 batch：

```bash
python clean_object_names_qwen.py \
  --root /share/project/liuyou/humancoin/data/guanglun_depth \
  --start 1 --end 39 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --report-dir /share/project/liuyou/humancoin/data/guanglun_object_cleaning_reports \
  --model qwen3.7-plus --workers 4 --apply
```

简智待发布的 71 个 batch：

```bash
python clean_object_names_qwen.py \
  --root /share/project/liuyou/humancoin/data/jianzhi/wait_for_publish \
  --start 1 --end 71 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --report-dir /share/project/liuyou/humancoin/data/jianzhi/object_cleaning_reports \
  --model qwen3.7-plus --workers 4 --apply
```

`--apply` 才写回。如果仅想检查读取范围，用 `--limit 1 --dry-run` 替换 `--apply`，不会调用模型。如果去掉 `--apply` 且不加 `--dry-run`，则调用模型生成名称映射但不改标注，之后用相同报告目录加 `--apply` 会复用结果。

其他常用参数：`--batches batch_0000001 batch_0000003` 选择不连续批次；`--limit 10` 限制样本数；`--timeout 180` 设置请求超时；`--max-tokens 4096` 设置输出上限。不指定批次范围时处理根目录下所有 batch。

## 报告与恢复

- `results.jsonl`：逐样本修改前后名称、原因及状态。
- `summary.json`：处理文件数、修改名称数、待复核数及表外名称。
- `unresolved_names.json` / `unresolved_names.csv`：保留原值的待复核项。
- `new_names.json`：不在 Excel 参考词表中的规范名称，单独记录，不自动写入词表。
- `errors.json`：失败文件。
- `jobs/`：模型回执、原始响应、缓存映射和写回前标注副本。

报告目录必须在数据根目录之外。每次运行更新本次汇总，保留历史缓存和备份。同一输入、模型、词表及提示规则下复用模型结果，写回后重复运行也不会再次为同一映射调用模型。超时或不明确的请求结果不会自动重试；检查后可用 `--retry-failed` 允许重新请求。

写回前检查完整原文件未变化，仅替换 `objects` 数组各项 `name` 的 JSON 值；保留其他字节、排版和数组结构。所有更改均经语义及字节范围约束核对。

## 验证状态

7 项本地测试通过，覆盖仅改名称、重复项保留、量词修饰校验、缓存、旧缓存恢复、单项待复核及未知网络结果保护。光轮、简智各一个真实样本读取预检通过；工具实现阶段未实际调用模型或批量修改服务器标注。

```bash
python -m unittest discover -s tests/readme_upload -p test_object_name_cleaning.py -v
```
