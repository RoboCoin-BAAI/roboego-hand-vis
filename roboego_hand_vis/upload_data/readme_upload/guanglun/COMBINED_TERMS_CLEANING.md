# 物品名称与动作词表：统一生成、延后替换

统一入口：[clean_names_and_actions_qwen.py](clean_names_and_actions_qwen.py)。复用现有两个 Qwen 3.7 清洗器，逐样本生成物品名称映射和动作词表映射，最终汇总为一个独立的 JSON 结果文件。

## 两阶段流程

1. `prepare`：只读原标注，调用模型，保存一个替换结果文件。可以与场景清洗同时进行，因为它不会写原标注。
2. 等场景清洗结束后运行 `apply`：不调用模型，读取最新的原标注，只替换 `objects[].name`、`actions.vocabulary`，保留已经清洗好的场景及其他内容。

结果文件按每份标注的相对路径及物品索引记录原值、新值、映射依据和待复核项，不做全文件字符串替换，也不保存一份旧标注再整体覆盖。因此 `task_description`、动作原文、子任务的 `action` / `object` / `destination`、属性和场景分类均不会被替换。

## 服务器命令：简智

```bash
export PATH=/share/project/liuyou/humancoin/envs/ego/bin:$PATH
cd /share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools
```

先生成一个独立结果文件（不修改原标注）：

```bash
python clean_names_and_actions_qwen.py prepare \
  --root /share/project/liuyou/humancoin/data/jianzhi/wait_for_publish \
  --start 1 --end 71 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --output /share/project/liuyou/humancoin/data/jianzhi/cleaned_terms.json \
  --model qwen3.7-plus --workers 4
```

场景清洗完成后，再执行替换：

```bash
python clean_names_and_actions_qwen.py apply \
  --bundle /share/project/liuyou/humancoin/data/jianzhi/cleaned_terms.json \
  --report-dir /share/project/liuyou/humancoin/data/jianzhi/terms_apply_reports
```

## 服务器命令：光轮

```bash
python clean_names_and_actions_qwen.py prepare \
  --root /share/project/liuyou/humancoin/data/guanglun_depth \
  --start 1 --end 39 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --output /share/project/liuyou/humancoin/data/guanglun_cleaned_terms.json \
  --model qwen3.7-plus --workers 4
```

等场景清洗完成后：

```bash
python clean_names_and_actions_qwen.py apply \
  --bundle /share/project/liuyou/humancoin/data/guanglun_cleaned_terms.json \
  --report-dir /share/project/liuyou/humancoin/data/guanglun_terms_apply_reports
```

## 结果文件及运行状态

`--output` 指定的一个 JSON 文件包含：

- 数据根目录、模型和处理状态。
- `records[].path`：样本标注的相对路径。
- `before`：原物品名称数组、原动作词表及用于判断的任务描述。
- `after`：清洗后的物品名称数组、清洗后的去重动作词表。
- 名称及动作的映射依据、表外词及待复核项。
- 失败样本及总体统计。

物品数组数量和顺序保持不变，即使清洗后同名也不合并；动作词表按首次出现顺序合并去重。原子任务标签不变，这是本次明确选择的清洗范围。

默认另有 `<输出文件名去后缀>_state/` 运行目录，保存模型回执、缓存和进度，属于运行状态，不是需要应用的结果。也可用 `--state-dir` 指定。最终 `apply` 只需结果 JSON 和工具代码，不依赖模型密钥或运行缓存。

`prepare` 复用两个既有清洗器，每个样本分别处理两个字段，沿用模型缓存，不会重复调用已完成的请求。中断后用相同命令重新生成结果，已有完成响应可以复用；失败或计费结果未知的请求默认不自动重试。确认后用 `--retry-failed` 允许再次调用。

`prepare --limit 1 --dry-run` 可只验证读取及结果格式，无模型调用；这类预览结果禁止应用。生成中的结果也禁止应用，只有最终状态为 `complete` 才能进入 `apply`。若部分样本失败，应用时跳过这些样本并在报告中标错，其他完成样本仍可替换。

## 保留最新场景与冲突检查

应用时读取当前文件，对比物品名称数量、顺序、原值、原动作词表和任务描述。场景分类与其他无关字段可以变化，这些最新值会保留。若目标字段已被其他程序改动，且既不是记录的原值也不是本次目标值，则拒绝旧映射并记入 `apply_report.json`。

同一结果可以重复应用，已经是目标值的文件跳过。应用前在报告目录 `original_annotations/` 下保存当前原文件副本；两个字段一起生成最终内容后原子写回。未增加 JSON 字段，也不会用生成结果时的旧场景覆盖当前场景。

**`apply` 必须等场景清洗写回结束后运行**，也不要与原来的物品、动作脚本同时写同一批文件。`prepare` 可以与场景清洗并行；若恰逢场景写入触发读取冲突，程序会先尝试复用已保存的响应重新读取，仍失败的样本可在下次运行恢复。

## 验证

2026-11-18 修复了动作响应的顶层包装兼容问题：模型返回 `[{index,name,reason}, ...]` 时也会解析为映射数组，仍严格检查索引、数量和名称，不放宽内容校验。已兼容修复前的动作缓存身份，旧 `response.json` 可直接复用；终端进度新增具体 `error` 内容。遇到旧日志中的 `Expected exactly one indexed mapping per original label` 时，停止旧进程并用同一 `--output`、`--state-dir` 重跑即可，不要删除缓存，也无需为包装格式问题添加 `--retry-failed`。

物品解析也兼容旧缓存。对于数量、顺序和字段结构正确、但个别候选名称仍含修饰或缺少解释的响应，只将该项标为待复核并保留原名，其他合法名称及动作映射可以继续处理；数量/索引不符、响应不是有效 JSON 等结构错误仍会报错，不猜测对应关系。

4 项合并流程测试通过，覆盖：准备阶段不写回、场景已变化时保留最新值、目标字段冲突拒绝替换、预览结果禁止应用以及重复应用跳过。简智一个真实样本的无模型调用预检通过。本次尚未调用模型或批量修改标注。

```bash
python -m unittest discover -s tests/readme_upload -p test_combined_cleaning_bundle.py -v
```
