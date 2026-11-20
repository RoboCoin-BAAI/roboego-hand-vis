# 动作词表清洗

入口：[clean_action_vocabulary_qwen.py](clean_action_vocabulary_qwen.py)，默认 `qwen3.7-plus`，Python 3.10+。参考词表为 [action_vocabulary.json](action_vocabulary.json)，从 `D:/data/annotation_statistics/理想/HumanCOIN_Subtask_Atomic.xlsx` 导出。

## 范围与规则

按用户确认，**只清洗 `actions.vocabulary`**。不修改 `subtask[].action`、动作原文、时间、物体或其他字段；也不从子任务标签重新构建词表。因此词表规范化后，子任务里仍可能保留 `retrim` 等原标签，这是此次限定范围的结果。

- 去掉形容词、副词、执行次数、方向、物体名等修饰，将动作短语简化为基础动词。
- 优先使用 Subtask_100 和 Atomic_50 中的词，合并同义标签，按首次出现顺序去重。
- 例如 `retrim → trim`、`repeatedly trim hair → trim`、`carefully wipe the table → wipe`、`pick up → pick`。
- 不机械删除词头，不把 `remove` 当成 `move`；不合并 `open/close`、`fold/unfold`、`screw/unscrew`、`attach/detach` 等不同或相反动作。
- 表内已经存在的 `adjust_grip` 可接受，但含义相同时优先使用更简洁的 `adjust`。
- 词表是优先参考，不是封闭枚举。没有合适参考词时可保留忠实、简洁的基础动词，单独列入表外词清单。
- 无法判断的标签保留原文并列入待复核清单；非法输出不写回原文件。

示例：

```json
{"vocabulary": ["trim", "check", "adjust", "retrim"]}
```

归一化后：

```json
{"vocabulary": ["trim", "check", "adjust"]}
```

Excel 两张表末尾均存在无效数字 `110`，导出时排除。有效参考词实际为 **99 个 Subtask 动词、46 个 Atomic 动词**，不为了凑表名中的 100/50 而自行补词。

## 运行命令

工具已同步到 `/share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools/`，与前两个清洗脚本共享 `apply_scene_classification.py`、`reclassify_scenes_qwen.py` 及 Argus 的网关配置。需要已配置的模型密钥，不需要抽帧或读取视频。

```bash
export PATH=/share/project/liuyou/humancoin/envs/ego/bin:$PATH
cd /share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools
```

光轮前 39 个 batch：

```bash
python clean_action_vocabulary_qwen.py \
  --root /share/project/liuyou/humancoin/data/guanglun_depth \
  --start 1 --end 39 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --report-dir /share/project/liuyou/humancoin/data/guanglun_action_cleaning_reports \
  --model qwen3.7-plus --workers 4 --apply
```

简智前 71 个 batch：

```bash
python clean_action_vocabulary_qwen.py \
  --root /share/project/liuyou/humancoin/data/jianzhi/wait_for_publish \
  --start 1 --end 71 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --report-dir /share/project/liuyou/humancoin/data/jianzhi/action_cleaning_reports \
  --model qwen3.7-plus --workers 4 --apply
```

用 `--limit 1 --dry-run` 替换 `--apply` 可只做读取预检，不调用模型。不加 `--apply` 且不加 `--dry-run` 时调用模型生成映射报告，之后在同一报告目录加 `--apply` 会复用缓存结果。写回后再次运行也会识别已应用的去重结果，不重复调用。

## 输出与保护

- `results.jsonl`：逐文件原词表、新词表、映射依据及状态。
- `summary.json`：文件处理、写回、待复核数量和表外动词。
- `unresolved_labels.json` / `unresolved_labels.csv`：保留原值的待复核标签。
- `new_verbs.json`：不在参考词表中的动词。
- `errors.json`：处理失败文件。
- `jobs/`：模型回执、原始响应、映射缓存和写回前的文件副本。

报告与备份目录须在数据根目录之外。每次运行覆盖本次汇总，但保留历史缓存和原文件副本。超时或未知网络结果不会自动重试，确认后可用 `--retry-failed` 明确允许再次调用。

写回前检查完整原文件没有发生并发变化，仅替换 `actions.vocabulary` 对应的 JSON 数组；该数组内部会因去重和序列化改变排版，其他字节保持原样。不要与场景清洗、物体名称清洗对同一文件同时运行 `--apply`；可以并行生成报告，再串行写回。

## 验证状态

7 项本地测试通过，验证字段范围、去重顺序、相反动作保留、参考词导出、缓存复用、顶层数组响应及旧版本缓存恢复。光轮、简智各一个真实样本读取预检通过。工具实现阶段未实际调用模型或批量修改数据；后续用户运行产生的已保存响应可用于离线复查。

```bash
python -m unittest discover -s tests/readme_upload -p test_action_vocabulary_cleaning.py -v
```
