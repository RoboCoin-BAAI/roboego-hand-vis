# 物品、动作及任务名称统一更新

入口：`run_annotation_updates.py`，Python 3.10+，模型默认 `qwen3.7-plus`。复用已运行的物品、动作清洗结果；不会重新调用物品、动作模型。任务名和共用任务名调用模型生成。

## 更新范围

1. 从 `prepare_progress.jsonl` 固定所选批次的清洗结果，更新 `objects[].name` 和 `actions.vocabulary`。
2. 对成功写回的样本生成 `task_name`：Subtask 词表中的动词 + 主物品。判断不确定或不满足词表约束时保留原值。
3. 根据描述、子任务及当前任务名，为所有所选样本添加 `common_task_name`，例如 `object_cleaning`、`hair_styling`。不确定时用 `other`。

只改上述四个位置；不改场景、子任务动作/物品、属性或其他字段。备份和结果位于发布目录外的 `--report-dir`。无需提前检查视频。

## 服务器命令

先进入已配置模型 API 环境的终端：

```bash
export PATH=/share/project/liuyou/humancoin/envs/ego/bin:$PATH
cd /share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools

# 简智：1–35 已处理，剩余从 36 开始
python run_annotation_updates.py \
  --dataset jianzhi \
  --root /share/project/liuyou/humancoin/data/jianzhi/wait_for_publish \
  --start 36 --end 71 \
  --terms-state-dir /share/project/liuyou/humancoin/data/jianzhi/cleaned_terms_state \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --report-dir /share/project/liuyou/humancoin/data/jianzhi/annotation_updates_36_71 \
  --model qwen3.7-plus --workers 4 --apply

# 光轮
python run_annotation_updates.py \
  --dataset guanglun \
  --root /share/project/liuyou/humancoin/data/guanglun_depth \
  --start 36 --end 100 \
  --terms-state-dir /share/project/liuyou/humancoin/data/guanglun_cleaned_terms_state \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --report-dir /share/project/liuyou/humancoin/data/guanglun_annotation_updates_36_100 \
  --model qwen3.7-plus --workers 4 --apply
```

去掉 `--apply` 可只预检，不调用模型、不写标注、不创建报告。如果所选范围仍有样本未生成清洗记录，会在写回前退出；等待清洗完成或缩小到已全部完成的批次。

如果清洗分多次运行，同一状态目录的总进度文件可能只剩最后一次的范围。此时可加 `--recover-cached-terms`，从原始物品、动作缓存恢复记录，不请求物品、动作模型。配合 `--apply` 时将恢复记录保存在报告目录的 `terms_cache_recovery.json`，不覆盖清洗生产任务的进度文件。真正缺少缓存的样本仍会阻止写回。单独预检恢复时可能更新已有缓存目录中的解析结果，但不修改标注或创建报告。

## 恢复与异常

- 中断后原命令、原报告目录继续，已完成的生成和写回跳过。报告目录绑定数据集、范围及分类规则，不可换范围复用。
- 技术错误记录在 `pending.json`。处理原因后，在原命令末尾加 `--retry-failed`，显式允许失败模型请求再次计费。成功请求不重跑。
- 技术性的详细任务名错误解决后才进入共用任务名阶段；语义复核项不会阻塞。
- 物品/动作准备失败的记录被固定在快照中：保留原标签，不生成详细任务名，但仍可生成共用任务名。它们仍列入 `pending.json`，退出码为 1；修复准备结果后应单独处理这些样本，不要对整个范围重新建报告运行。
- 详细任务名复核见 `task_review.json`；共用任务名 `other` 见 `common_other.json`；汇总见 `summary.json`。`other` 不建议自动合并。
- 已存在且不同的 `common_task_name` 会拒绝覆盖。不要让其他写标注的程序与本脚本同时运行；本入口的锁只协调本入口自身。
- 断点记录保存的是已成功的操作，不用于重新审计后续人工改动。

验证：用模拟模型测试完整顺序、字段保护、预检无写入、缺记录拒绝执行和中断恢复；未启动剩余批次的付费模型任务。
