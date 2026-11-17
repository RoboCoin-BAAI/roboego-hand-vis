# 结构化标注命名统计

先统计，再人工制定清洗映射。本工具只读原始 JSON，不进行自动归并或回写，仅依赖 Python 3.10+ 标准库。

在项目根目录执行（Linux 示例）：

```bash
python -m roboego_hand_vis.annotation_statistics.run \
  --input-root /share/project/liuyou/humancoin/data/guanglun_depth \
  --output-dir /share/project/liuyou/humancoin/annotation_statistics/guanglun_depth
```

按光轮固定层级 `<数据集根目录>/batch_*/<session>/annotation/episode_000000_structured.json` 查找，仅列举 batch 和 session 目录，不进行全目录递归，不进入 videos、data 或 chunk 目录。

输入也可以是单个 `batch_*`、单个 session 或 annotation 目录，分别直接从对应层级查找。不支持在数据集根目录以上增加任意层级。要统计其他 episode 编号，增加 `--pattern 'episode_*_structured.json'`。

输出：

- `summary.json`：匹配数、读取成功/失败数、字段异常数、各字段不同原值数量、空值/缺失文件数。
- `task_name_counts.csv`、`scene_counts.csv`：原值、出现次数、覆盖文件数。
- `common_task_counts.csv`：统一任务名、出现次数和覆盖文件数；另生成 `common_task_mapping.csv`。缺失、null 和空白文件数记录在 `summary.json` 的 `fields.common_task.missing_or_blank_files`。
- `object_name_counts.csv`：统计 `objects[].name` 的原始名称、出现次数和覆盖文件数；同一文件内同名对象重复出现时，出现次数累计，覆盖文件数仅计一次。另生成 `object_name_mapping.csv` 供后续规范名称。
- `category_counts.csv`、`area_counts.csv`：分别统计 `scene_classification.category` 和 `scene_classification.area`，不组合、不从 scene 描述推断。缺失/null/空白数量分别记录在 summary 中。
- `action_counts.csv`：`subtask[].action` 的动作出现次数（重复子任务也计数）和覆盖文件数。
- `action_vocabulary_counts.csv`：单独统计 `actions.vocabulary`，不与子任务动作叠加。
- `episodes.jsonl`：每个成功读取文件的相对路径及上述八项原始值，便于溯源。
- `errors.csv`：无法读取、JSON 错误或字段类型异常的位置。存在错误/字段异常时仍完成统计，命令退出码为 1；未匹配到文件也报错。
- `*_mapping.csv`：原名和待填写的 `canonical` 规范名，并附频次。当前阶段只生成表格，不应用映射；空白不代表删除。

按原始字符串精确计数，保留大小写、空格及空字符串差异，缺失/null 不计为名称。CSV 使用 UTF-8 BOM，可用 Excel 打开。`episode_count` 按标注文件计数。

建议先按频次查看，再填写 mapping 的 canonical 列，将同义原名映射到同一个规范名。统计可重复运行：报告会覆盖，已有 mapping 文件保留，以免覆盖人工编辑；输入范围变化时应使用新的输出目录生成匹配该范围的映射。

远程服务器数据需要在服务器运行本命令；本地创建工具不会自动访问远程路径。
