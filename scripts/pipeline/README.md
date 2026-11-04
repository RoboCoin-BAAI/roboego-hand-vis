# CPU: fixed range, one complete stage at a time

CPU freezes its session selection at startup and runs CPU stages in `run_stages` order.
L1: download ALL -> convert ALL -> QC ALL -> rectify ALL -> Argus ALL -> VIO ALL.
L3: download ALL -> QC ALL -> rectify ALL -> Argus ALL -> world hand points ALL.
`--workers` controls concurrency within each stage. Each stage drains the fixed
selection before the next stage starts. L3 uses `cpu_failure_policy: continue_passed`:
failed sessions are reported and excluded from all later CPU phases; successful
sessions continue. The final exit code is nonzero if any session failed, even when
all survivors finish. L1 retains the default `stop` policy: one failed session
blocks the next phase for the entire selection.

L3 `points` generates both world-space 21-point hands from MANUS + VIVE using the
confirmed mounting transform. Output hashes, source hashes and mounting parameters
are validated before success is recorded. `points` runs last and depends only on QC. Rectification depends on QC, and review depends on rectification; points failures do not block them. The default pose interpolation bracket is 30 ms; larger gaps produce NaN samples.
L3 review uses official `qwen3.8-27b` for labeling and structured completion,
1 FPS sampling, and `--structured-only`. Configure `DASHSCOPE_API_KEY` and
`DASHSCOPE_BASE_URL`, and unset `XMSXB_API_KEY`. VIO is available as an explicit
stage but is no longer in the default L3 CPU sequence.

L3 review now requires the episode's unique EXEC_START marker from
`meta/qc_reports.jsonl` (`qc_report.streams.marker.stats.markers.items[].t_offset`).
Only execution frames at/after that offset are sent; annotation times remain on
the original RUN_START/video clock. Missing or invalid markers fail explicitly.
Sync Argus `review/L3/run.py`, `review/__main__.py`, and `label/episode.py` together.
To replace earlier full-video annotations after configuring the API environment:

```bash
bash scripts/L3/pipeline_cpu.sh --batch-range 1:9 --workers 4 \
  --run-stages review --force-stages review
```

Force archives the previous annotation and makes new model calls. QC marker
changes are included in the review stage's input signature.

```bash
# L1 batches 1 through 12, inclusive
bash scripts/L1/pipeline_cpu.sh --batch-range 1:12 --workers 4
# L3 explicit batch selection
bash scripts/L3/pipeline_cpu.sh --batches batch_0000001 batch_0000003 --workers 4
# Resume after fixing failures; valid completed results are skipped
bash scripts/L3/pipeline_cpu.sh --batch-range 1:12 --workers 4 --retry-failed
```

`--batch-range` is CPU only and cannot be combined with `--batches`. Without either,
CPU selects all batches discovered at startup. CPU `--watch` waits for externally
locked work within that fixed selection; it does not discover new sessions.
Failed-session handling follows `cpu_failure_policy`. `--run-stages` must respect dependency order; omitted
prerequisites still require valid existing state. Use `--dry-run` to inspect the
selection without launching work. Existing `--force-stages`, state validation,
backup/restore, and dependency checks remain in effect.

GPU retains its independent script, index discovery and model scheduling.
The older watch/discovery notes below apply to GPU; CPU follows the rules above.

---

# CPU / GPU 独立流水线

本目录是 L1、L3 共用调度器。L1/L3 的 pipeline.yaml 注册各自的数据路径、阶段命令及依赖。
将来接入 jianzhi、tashi 时新增配置及数据加载适配，不复制调度器。
手部阶段 hand 已预留但 enabled: false，未接入真实命令，不会运行。

## 两台机器运行

两台 Linux 机器需共享数据目录及 paths.state_root，并使用同一 dataset 标识、阶段定义和 start_at。
共享文件系统必须支持跨主机 POSIX flock；只共享数据、不共享状态无法防止重复领取。
同一 session 的阶段互斥，不同 session 可并行；GPU 每个 ID 同时处理一个任务。
暂不支持同卡同时驻留深度和手部模型，后续按显存和吞吐测试再扩展。

```bash
# CPU 机器，L1 含转换；L3 没有转换
bash scripts/L1/pipeline_cpu.sh --batches batch_0000001 --workers 4
bash scripts/L3/pipeline_cpu.sh --batches batch_0000001 --workers 4

# 对应的 GPU 机器；持续等待 CPU 发布合格状态
bash scripts/L3/pipeline_gpu.sh --batches batch_0000001 --gpus 0,1,2,3 --watch
```

不指定 --batches 则读取 input_root 全部 batch CSV。--watch 定期重新扫描 CSV，Ctrl+C 停止等待。
不加 --watch 时，任务暂不可运行/失败会列入 incomplete 并以非零退出，便于发现阻塞。
PIPELINE_CONFIG 覆盖 YAML 路径，PIPELINE_PYTHON 覆盖启动解释器；YAML 的 paths.python 是实际子任务解释器。
先设置好模型接口环境变量、依赖与 OpenVINS 二进制，新调度器不自动 apt 安装或编译。

CPU L1：download → convert → qc → rectify → review → vio。
CPU L3：download → qc → points → rectify → review。
GPU：depth、hand 都要求 qc 和 rectify 有效完成，不能绕过门禁。
下载按 session 执行；配置的输出仍为 batch/session。

## 选择阶段与覆盖

```yaml
start_at: download
run_stages: [download, qc, rectify, review, points, depth]  # L3；depth 为 GPU 阶段
force_stages: []
```

```bash
# 仅在 GPU 重跑深度，成功后替换，保留旧结果和执行历史
bash scripts/L3/pipeline_gpu.sh --batches batch_0000001 \
  --run-stages depth --force-stages depth --gpus 0,1

# CPU 重跑 QC 和去畸变；已有下游状态立即失效
bash scripts/L3/pipeline_cpu.sh --batches batch_0000001 \
  --run-stages qc,rectify --force-stages qc,rectify --workers 4
```

force_stages 必须包含在 run_stages 中，各端只运行本端负责的阶段。
强制请求在本次进程内只执行一次，--watch 不会不断重跑同一个成功或失败任务。
失败默认不自动重试；下一次加 --retry-failed 或显式 force 才会重试。
付费标注同样不自动重试。多机器使用 force 时只在一台机器发起，避免两次独立强制请求。
已有记录的输入/产物变化会失效；不再把旧文件存在视为当前执行成功。

## 已下载、已转换、已去畸变的数据

将两端 YAML 的 start_at 都设为 qc，省去下载/转换，不更改 qc→rectify→下游的门禁。
首次 CPU 登记可执行：

```bash
PIPELINE_CONFIG=/path/existing_l3.yaml bash scripts/L3/pipeline_cpu.sh \
  --batches batch_0000001 --run-stages qc,rectify --adopt-existing
```

首次无记录时，--adopt-existing 验证已有 QC 报告通过、已有矫正产物和当前标定/视频匹配后登记。
不存在或验证不通过则实际运行对应模块。已有失败或过期记录不会重新导入旧报告蒙混放行。
导入旧 QC 报告是显式信任它所对应的采集数据；若数据已修复或报告来源不确定，请实际重跑 QC。
不直接导入旧 Bash 的 .done 标记，也不通过手改 passed 来同步状态。
若从原始下载开始，L1 下载和转换依赖会照常建立；start_at: convert 允许从已下载的 L1 开始。

## 记录与恢复

状态目录：state_root/dataset/batch/session/。

- state.json：各阶段当前状态、依赖代次、输入签名、产物快照和最后日志。
- runs/<id>/command.json、console.log、result.json：每次命令、日志、结果。
- runs/<id>/previous：强制覆盖前备份；失败恢复原位置，失败产物留作排查。
- session.lock：OS 文件锁；锁随进程退出释放，不按超时时间抢占仍在运行的任务。

CPU 和 GPU 的并发以 session 为单位，避免 CPU 改元数据时 GPU 同时使用同一数据。
任务状态存的是 batch/session，产物路径根据本机 paths 重建；两端路径可不同，但必须指向同一份数据和状态。
输入/产物签名使用配置、依赖代次及文件大小/mtime，不是逐文件内容哈希；
保留大小和 mtime 的人工替换不能自动发现。代码或模型内容更新后请显式 force 对应阶段。
旧文件恢复不代表状态通过：失败阶段仍为 failed，下游仍被阻断。

测试覆盖 CPU/GPU 门禁、上游失效、强制失败回滚、失败不循环重试、配置环路和未来数据集路径。
Windows 测试替代了 Linux 文件锁接口；真实跨主机锁及 GPU/OpenVINS 后端需服务器验证。
本次开发没有运行实际下载、模型推理或付费标注。

原 scripts/old/L1/run.sh、scripts/old/L3/run.sh 是旧的单机整批入口，保留但不要与新调度器同时处理同一数据。


## Rectification completion index

The CPU pipeline maintains `output_root/batch_*/rectification_done.json`: a JSON array
of absolute session paths, one entry per successfully validated session. For
example, L1_lerobot/batch_0000001/rectification_done.json includes paths from all processed
batches. Repeated runs merge entries and remove duplicates. Before rerunning
upstream stages, a session is removed; successful rectification adds it back.
Updates use a shared lock and atomic replacement.

GPU startup/watch reads this file instead of rediscovering tasks from batch CSVs.
`--batches` filters paths by their batch directory. Existing state/dependency and
artifact checks remain in effect; listing a directory alone does not bypass QC.
No index means no ready GPU tasks (use --watch to wait for CPU output).
For prior completed pipeline runs, run the CPU entry again with the same batch
selection: current successful rectification states are registered without
recomputing. This index is maintained by the shared pipeline, not by standalone
rectification commands; standalone output still needs pipeline validation/adoption.


## Eight-GPU staged inference

GPU defaults to IDs 0,1,2,3,4,5,6,7. Each JSON snapshot drains depth first,
with two session jobs/model instances per card (up to 16 sessions). Only after
all runnable depth tasks have finished does hand inference start, one job per
card (up to 8 sessions). Failed tasks are reported, not endlessly retried, and
remain subject to their configured downstream dependencies. Depth and hand
phases do not overlap within one runner. With --watch, newly indexed sessions
are picked up after the current snapshot's phases finish.

```bash
bash scripts/L1/pipeline_gpu.sh --gpus 0,1,2,3,4,5,6,7 --depth-workers-per-gpu 2
```

Depth and hand now use resident JSON-lines workers. Each lane loads its model
once, handles multiple sessions, and exits at the phase boundary. Per-session
state, output checks, logs and failures are still written by the scheduler.
The first task log records model_load_s. With --watch idle polls retain the most recently active phase workers.
New tasks for the same phase reuse loaded models; switching between depth and
hand releases the previous phase first. With both stages selected, newly arrived
depth tasks therefore require a phase switch if hand models are resident. L3 currently configures depth only, without a hand stage.
Two instances per A800 80GB is the requested concurrency, not a measured memory
or throughput guarantee. Avoid multiple runners sharing the same GPU allocation.

When start_at is qc, CPU discovery reads output_root/batch_*/session/meta/info.json directly (shallow scan), without batch CSVs. --batches filters these batch directories. Download/convert startup continues to use input CSVs.

Batch-local indices: CPU updates only the current batch file under output_root/batch_N/rectification_done.json. GPU --batches reads exactly those files; without --batches it reads all batch_*/rectification_done.json files. The legacy root index is no longer read. Each file keeps atomic writes, locking and deduplication. Restart existing runners to load the new code.
