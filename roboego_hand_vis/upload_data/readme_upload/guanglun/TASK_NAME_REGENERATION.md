# 简智任务名重新生成

入口：[regenerate_task_names_qwen.py](regenerate_task_names_qwen.py)。先完成物品和动作词表清洗，**并把清洗结果应用到标注文件**，再运行本脚本。2026-11-18 已启动简智、光轮 1–35 的模型生成及写回任务，仅选择物品/动作成功写回的样本。

**1–35 已完成，执行结果、合并索引和备份说明见 [TASK_NAMES_01_35_RESULT.md](TASK_NAMES_01_35_RESULT.md)。**

## 规则

- 格式为小写英文 `Subtask 动词 + 主物品类别`，例如 `cut hair`、`fold towel`、`wash cup`、`wipe table`。
- 动词严格使用 `action_vocabulary.json` 中的 `Subtask_100` 组，不能新增动词或使用仅属于 Atomic 的词。
- 根据整段任务目的选择动词，参考 `task_description`、`original_description`、子任务以及已清洗的动作词表。理发任务使用 `cut hair`，不因为片段中多次出现 trim 就归为 `trim hair`。
- 主物品以已清洗的 `objects[].name` 为证据，可以使用有证据的上位类别或规范同义词，例如多种衣物共同整理归为 `organize clothing`，不任取 `hoodie` 代表所有衣物。模型返回动词、物品类别、支持该类别的物品索引及理由，脚本校验后拼接任务名；不修改物品字段。
- 主物品明确出现在描述中时，也可以直接以描述为证据，不要求背景目标（例如桌子）必须单列在物品数组中。索引形式错误不能单独推翻明确的文字证据；没有任何目标证据的候选不会写回。
- 普通整段理发统一 `cut hair`，明确局部修整、刘海、分叉发尾等仍可用 `trim hair`。标识用途的贴纸可统一 `stick label`，装饰贴纸保留 `stick sticker`。上述是任务名的归并规则，不改物品词表。
- 动词不要求已经存在于 `actions.vocabulary`，后者可能只有 Atomic 级操作。
- 任务目的冲突、缺少主物品或无法找到忠实的 Subtask 动词时，记录为 `review`，保留旧任务名。待处理明细写入 `<output_stem>_state/review.json`。
- `prepare` 只生成独立结果，`apply` 只修改顶层 `task_name`；原本缺少时仅补上这个字段。保留场景、物品、动作、子任务、时间和其余字节，不添加其他字段。
- 允许将模型误填在动词栏的完整同名短语规整为动词，例如 `verb="cut hair"`、`object_name="hair"` 可恢复为 `cut hair`。无法通过词表或物品索引校验的完整响应转为待复核，不强行写回；格式损坏或截断响应仍记为错误。
- 为后续合并使用 `(scene_classification.category, scene_classification.area, task_name)` 三个字段作为分组键。不确定样本应先完成复核，避免按其旧任务名自动合并。

## 部署与运行

将本脚本放入服务器现有的 `guanglun_scene_qwen_tools`。它依赖同目录的 `reclassify_scenes_qwen.py`、`apply_scene_classification.py`、`action_vocabulary.json`，并复用 Argus 的 Qwen 端点配置。默认模型为 `qwen3.7-plus`，使用现有 API 环境变量。

先用少量样本生成结果，确认后去掉 `--limit 10` 处理 1–71 全部样本：

```bash
export PATH=/share/project/liuyou/humancoin/envs/ego/bin:$PATH
cd /share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools

python regenerate_task_names_qwen.py prepare \
  --root /share/project/liuyou/humancoin/data/jianzhi/wait_for_publish \
  --start 1 --end 71 --limit 10 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --terms-apply-report /share/project/liuyou/humancoin/data/jianzhi/terms_apply_01_35_20261118/apply_report.json \
  --output /share/project/liuyou/humancoin/data/jianzhi/task_names.json \
  --model qwen3.7-plus --workers 4
```

`--dry-run` 仅预览选中的样本，不调用模型，生成的预览文件不能用于 `apply`。正常生成会输出逐样本进度，并把完整记录持续写入状态目录的 `progress.jsonl`；完成后才把结果文件标记为 `complete`。

`--terms-apply-report` 根据物品/动作写回报告筛选成功的样本，不必重新遍历全量目录；省略时按批次目录发现样本。当前 1–35 选中简智 3,457 条、光轮 3,147 条；物品/动作准备失败的 58 条不在本次生成范围内。

确认独立结果后再写回（不调用模型）：

```bash
python regenerate_task_names_qwen.py apply \
  --root /share/project/liuyou/humancoin/data/jianzhi/wait_for_publish \
  --bundle /share/project/liuyou/humancoin/data/jianzhi/task_names.json \
  --report-dir /share/project/liuyou/humancoin/data/jianzhi/task_names_apply
```

写回前会重新读取最新标注：场景变化可以保留；物品名称、动作词表或任务证据变化则拒绝旧结果，必须重新生成。原文件备份到 `--report-dir/original_annotations/<batch>/<sample>/annotation/`，同名已有备份不会覆盖。已应用的任务名再次执行会跳过。

**不要与其他标注写入任务同时运行 `apply`。** 生成时也应等物品和动作清洗写回结束，否则结果可能失效。

模型响应按样本、证据、模型、提示和词表缓存；相同输入再次生成会复用响应。上次请求失败且可能已计费时，默认停止该样本，检查状态后可显式指定 `--retry-failed`。改变物品或动作词表后，会生成新缓存并重新请求模型。
