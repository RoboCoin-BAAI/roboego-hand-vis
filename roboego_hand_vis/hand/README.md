# Native Hand Launcher

Thin wrapper: MINT inference, followed by EgoHandKit's existing
`scripts/run_without_front.sh mint VIDEO PREDICTION DEPTH OUTPUT`.
No Docker, VIO, depth estimation, Ray or dependency installation is involved.

## Configure Once

Edit `recommended.yaml` or pass `--config /absolute/path/to/local.yaml`.
Set the two source checkouts and absolute Python executable paths (`~` works).
`mint_checkpoint` is relative to the MINT checkout, or absolute. The input video
and depth paths are relative to `--input_dir`; default depth is `depth/`.
For a session using `depth/fast_foundation`, configure that exact directory.
No automatic guessing or upstream data generation is performed.

MINT uses the existing CLI and prediction contract. Record/transfer its local
changes separately: a branch name alone does not capture uncommitted fixes.
EgoHandKit uses `feature/mint-frontend` with the native shell configuration patch,
not the Ray branch. The example checkout is `~/EgoHandKit`; MINT is
`~/wuji-ego-mint`. A worktree does NOT provision model assets or environments.

The outer Python needs PyYAML; algorithms use the configured interpreters.
Artifact checks use NumPy, OpenCV and PyArrow in the EgoHandKit environment.
CUDA/EGL settings are inherited, not repaired. Verify NVIDIA EGL on the native
host separately; Docker's environment fix is not automatically transferred.

## Command

From the native-hand project worktree, after configuration:

```bash
python -m roboego_hand_vis.hand.run \
  --input_dir /path/to/session \
  --output_dir /path/to/new-result \
  --dry-run
```

Remove `--dry-run` to execute. Dry-run only prints commands, cwd and environment
overrides: it does not validate model environments, create outputs, copy an
exporter or run inference. Real execution checks paths/interpreters and exporter
compatibility before inference and requires a NEW output directory unless
`--resume` is explicitly supplied. Dry-run shows the initial attempt paths;
an actual resumed run chooses paths from its checkpoint.

MINT retains chunked mode, max-frames 10000, target-fps 30, window 32,
window-batch-size 1, no-render. This is not an arbitrary-FPS/long-video adapter;
video/depth alignment must satisfy the existing algorithms.
EgoHandKit still owns all its arguments, including wrist surface compensation
and orientation gating. We do not substitute the older Docker profile.
The external script creates a missing exporter in MINT's scripts directory,
but refuses to overwrite a different existing copy. This explicit side effect
is retained; a full run is not read-only with respect to the MINT checkout.

## Outputs

By default, a successful run contains exactly three delivery files:

```text
new-result/
  hand_tracking.parquet
  render.mp4
  run.log
```

The Parquet and video come from EgoHandKit's unique `mint/*/final/` result.
This is the final Parquet-driven render, not a stage preview. MINT NPZ arrays
must match the exporter's required keys, shapes and input frame count. Final
checks read Parquet batches and compare row count and video metadata frame
counts with the input. They do not certify video pixel integrity, pose accuracy
or visual quality; `quality_status` remains `not_validated`.

`run.log` combines both subprocesses' stdout/stderr with timestamped launcher
events: commands, interpreter paths, source revisions/dirty status, run identity,
stage start/exit/validation, duration, resume decisions, publication hashes,
final summary, errors and cleanup. Resumes append rather than truncate it.
No full environment dump or separate report JSON is delivered.

Add `--full-output` to retain all attempts alongside the same three public files:

```text
new-result/
  hand_tracking.parquet
  render.mp4
  run.log
  .work/
    state.json
    attempts/
      mint-0001/prediction.npz
      egohandkit-0001/mint/...
```

Both modes first generate full intermediate outputs. Compaction reduces final
storage, not peak disk usage. It only deletes this run's `.work/` after verified
copies have been published and checkpointed. Inputs, weights and external
repository caches are never cleaned. Nested external symlinks are not followed
by deletion, and symlinked control paths/final artifacts are rejected.

## Resume

After a failure or interruption, rerun with the original arguments plus
`--resume` (include `--full-output` again if originally used):

```bash
python -m roboego_hand_vis.hand.run \
  --input_dir /path/to/session \
  --output_dir /path/to/interrupted-result \
  --resume
```

| Checkpoint | Recovery |
| --- | --- |
| MINT incomplete | Run MINT in a new attempt directory |
| MINT complete, EgoHandKit incomplete | Verify MINT artifact hash, skip MINT, restart EgoHandKit |
| Both complete, publication incomplete | Verify artifacts and publish, without inference |
| Publication complete, cleanup incomplete | Verify public file hashes and finish cleanup |
| Full-output run complete | Verify public file hashes and return without inference |
| Compact run complete | Refuse resume: its internal checkpoint has been removed |

There is no per-frame recovery or automatic retry. Failed attempts are retained
until successful compaction, and never reused as a partial stage result. A stage
is complete only after its command exits successfully, validation passes, and
its checkpoint is atomically written. SIGINT/SIGTERM terminates the active child
process group; failures/interruption retain available intermediates and the log.
Hard kills can leave children running: inherited locks prevent another launcher
from resuming while those children still hold the locks.

Before resuming inference, the launcher compares commands/paths/output mode,
source commit and source/config content, interpreter identities, selected runtime
environment settings, input video/depth, MINT checkpoint and known MANO/HaMeR asset
trees. Large data/model files use resolved path, size, modification and change
timestamps, not full content hashes; completed delivery artifacts use SHA-256.
This is not a snapshot of installed packages or arbitrary external dependencies.
Keep environments unchanged during recovery; use a new output for environment
upgrades, changed inputs or changed model/code. A matching auto-copied MINT
exporter is explicitly allowed because the wrapper owns that existing behavior.

After publication is checkpointed, cleanup recovery needs only the original
plan and public file hashes, not the input/model files or intermediate artifacts.
If cleanup stops after removing the checkpoint itself, valid public results
remain but resume is refused; an empty `.work/` may remain in this narrow window.

## Limits and Validation

EgoHandKit retains its original test_data cache. Per-checkout and per-output
advisory locks serialize this launcher for the same OS user; they do not guard
legacy/manual commands, other users or different checkouts sharing symlinked
caches. Do not run those concurrently. Separate output roots do not imply full
cache isolation. Existing Docker entrypoints are unchanged.

CPU tests use stand-in algorithms with real Parquet/NPZ/video artifacts and
subprocess execution. They cover compact/full delivery, restart boundaries,
identity changes, invalid artifacts, interrupted publication/cleanup and locks.
Install the project's `dev` extra for the test dependencies (including PyArrow).
No full native GPU run or visual quality acceptance is implied by these tests.

## L1 数据库入口

完成[数据库连接配置](../database/README.md)，并按本页说明配置 MINT、EgoHandKit 及各自 Python 环境后，在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m roboego_hand_vis.hand.database_run \
  --shared-root ./data/l1 \
  --limit 3
```

数据库入口使用当前环境的项目 `database` 依赖；算法及最终交付校验仍使用 `recommended.yaml` 中配置的独立 Python，不自动切换或安装环境。GPU 设置由子进程继承。

只领取下载、基础质检、视频粗质检、立体矫正、深度前门禁和深度估计均为 `completed`，且 `hand_status=pending` 的 L1 数据。必须等待 `fast_foundation_depth_status=completed`，不依赖可选手部采样检测或千问标注。省略 `--limit` 或设为 `0` 时处理当前所有可领取 session；`--config` 指定手部 YAML。

输入根目录由数据库 `local_path` 与共享根目录拼接；视频和逐帧深度按原配置读取，必须位于该 session 内。输出固定为每个 session 的 `hand/`：

```text
session/hand/
  hand_tracking.parquet
  render.mp4
  run.log
```

领取后写 `processing`。原 MINT → EgoHandKit 流程成功，最终 Parquet 行数、渲染视频帧数与输入视频帧数一致，交付文件校验通过后写 `completed`；执行异常、缺少输出或帧数不一致写 `failed`。此阶段不作为质量门禁，`quality_status` 仍为 `not_validated`，不自动写入 `block`。

数据库入口自动恢复已有 `.work/state.json`，校验原 checkpoint 的 plan 和 fingerprint；拒绝覆盖没有对应 checkpoint/数据库完成记录的已有 `hand/` 目录。失败记录不会自动重新领取，修正问题并将其 `hand_status` 从 `failed` 改回 `pending` 后，重新运行同一数据库命令。

完成记录存放在 `annotation/hand_database_receipt.json`，数据库运行及额外交付校验日志为 `annotation/hand_database.log`。记录保存 plan、输入/模型/代码 fingerprint、交付哈希和帧数校验；若计算已完成但数据库回写失败，重试可验证已有交付并补写状态，避免重复推理。输入、配置或交付文件发生变化时会拒绝复用。

默认最终输出仍为 compact。数据库模式先保留 native checkpoint，完成交付验证和数据库完成记录落盘之后，再清理 `.work/`；清理中断时可据完成记录继续恢复。加 `--full-output` 可保留中间结果，重试时保持原来的输出模式。

沿用原 launcher 的共享缓存锁：同一个 EgoHandKit checkout 需串行使用，不应同时启动多个手部数据库进程。数据库领取本身是并发安全的；推理期间不持有数据库事务。手部状态为 `completed` 后，深度归档数据库入口即可领取该数据。字段已存在，不需要数据库迁移。
