# 主手部操作任务名映射 V2（候选阶段）

每条任务名表达一个主要手部操作动词和一个主要操作物品概念。物品概念可以是多词，如 `hair dryer`。只生成外部映射建议，不写回标注。

判定依据是旧任务名及该名称下全部不同的任务描述和物品列表。这里只判断主要手部操作，不重新发明场景目标。输入来自 2026-11-19 的全量快照；没有重新读取全部子任务，因此描述不足或冲突的项目必须待复核。

## 规则

- 首选手部直接操作的主要目标物，不能用房间、目的地代替：搬放盒子到货架为 `organize box`，不是 `organize shelf`。
- 取放工具、操作结束后关门等辅助动作不进入任务名。
- 依据描述重点、重复操作判断主要动作。不能简单截取第一个动词或名词；对象或目标势均力敌时保留待复核。
- `wipe → clean` 继续作为任务名归并规则；保留 `wash` 与普通清洁的差异。
- 普通物品摆放语境下，可将 `arrange/tidy` 归为 `organize`；造型、装饰语境不能强制套用。
- `sort/stack/organize`、`pack/seal/wrap`、`cut/trim/shave`、`polish/clean` 仍区分。
- 物品按 V1 的显式单复数和同义词规则规范。模型必须给出原文物品片段与证据，程序核对片段存在且规范结果一致。
- 同一旧任务名下面有多种描述时，全部应支持同一映射；否则待复核，不能因为字符串相同就全局套用。
- 上下文过多的高频名称不截断证据，先转待复核。模型结果是候选，语法通过不等于人工认可。

## 示例

| 原内容 | 候选 |
| --- | --- |
| Wipe boxes and return cleaning tools | clean box |
| 将棋类盒逐个收入柜中并关门 | organize box |
| 将盒子上下堆叠 | stack box |
| 擦拭台面后归还海绵 | clean counter |
| 同等篇幅擦台面与镜子、无法确认主对象 | 待复核，不改成 clean bathroom |

## 运行

已准备完整的 3213 个不同旧任务名称及其全部不同描述。输入与策略保存在 `main_hand_task_mapping_20261119/prepared_inputs.json`、`manifest.json`。当前服务器连接没有可用的 API 环境，因此尚无模型合并结果。

在已配置模型的终端中：

```bash
export PATH=/share/project/liuyou/humancoin/envs/ego/bin:$PATH
cd /share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools
python review_main_hand_tasks.py \
  --inventory /share/project/liuyou/humancoin/data/task_names_inventory_20261119.json \
  --output /share/project/liuyou/humancoin/data/main_hand_task_mapping_20261119 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --model qwen3.7-plus --workers 4 --run-model
```

去掉 `--run-model` 只准备输入。脚本没有标注写回选项。

结果文件：`main_hand_task_mapping.csv`（原名、候选、状态、数据计数、理由），`results.json`（完整依据），`summary.json`（候选去重数与复核数量）。

同一命令可复用缓存继续，成功请求不重发；失败请求默认不自动重试，明确要重试时增加 `--retry-failed`。更改输入、模型或规则需新输出目录；API 失败原始响应保留，不能保证服务端未计费。

## 已有缓存逐条复核

校验已修正为逐条判定：单条证据或名词不合规则转待复核，不再使同组 8 条一起失败。结构损坏、截断、ID 或条数错误仍视为整组错误。已经运行的旧进程不会热更新，日志计数仍按旧逻辑。

可在运行中或运行结束后只复核缓存，不调用模型、不改运行任务的结果文件：

```bash
python review_main_hand_tasks.py \
  --inventory /share/project/liuyou/humancoin/data/task_names_inventory_20261119.json \
  --output /share/project/liuyou/humancoin/data/main_hand_task_mapping_20261119 \
  --audit-cache
```

独立输出 `cache_audit.json`、`cache_audit_mapping.csv`。`not_in_completed_cache` 是尚无完整响应的名称数量，不能直接当作失败数；运行结束后再执行一次可更新统计。此模式不需要 API 配置。
