# 光轮常驻深度消费者

从转换端的 `batch_*/rectification_done.json` 读取 KS3 前缀，只处理对应 `records/*.json` 中 `uploaded` 的对象。使用上传收据的相对文件名和大小下载完整 LeRobot 会话。输入侧与转换任务共用 `/share` 清单；两个流程分别使用 `runner.lock` 和 `depth_runner.lock`，可以同时运行。

```bash
cd /share/project/liuyou/humancoin/code/roboego-hand-vis
export PATH="/share/project/liuyou/humancoin/envs/ego/bin:$PATH"

python -u scripts/guanglun/depth.py \
  --state-root /share/project/liuyou/humancoin/data/guanglun_batches \
  --download-root /share/project/liuyou/humancoin/data/guanglun_depth \
  --model-root /share/project/liuyou/humancoin/code/Fast-FoundationStereo \
  --ckpt /share/project/liuyou/humancoin/code/Fast-FoundationStereo/weights/model_best_bp2_serialize.pth \
  --ks3-tool /share/project/liuyou/humancoin/utils/ks3/ks3util-linux-amd64 \
  --ks3-config /share/project/liuyou/humancoin/utils/ks3/ks3util-config-team \
  --gpus 0,1,2,3,4,5,6,7 --depth-workers-per-gpu 1 \
  --first-batch 1 --last-batch 574 \
  --execute
```

模型 checkpoint 旁边需要配套的 `cfg.yaml`，模型文件沿用 L1 的部署方式。新服务器必须能访问该共享目录，并有 CUDA、模型依赖和 ffmpeg/ffprobe。每卡默认一个常驻模型，可按显存改为 `--depth-workers-per-gpu 2`。首个会话下载完成后加载模型，此后跨会话、跨批次及空闲轮询期间都保持加载，不运行手部估计。

不带 `--execute` 只统计待处理条目，不下载、不加载模型。默认每 60 秒扫描一次并持续等待新增数据；`--once` 仅处理当前扫描到的条目后退出。批次按编号排列投递，多 GPU 会交错完成。下载和推理由各个 lane 顺序执行，不预先下载全部 574 批；队列最多容纳 lane 数量的两倍任务。

输出与恢复：

- 只有 resident 返回成功且深度完整性检查通过，才写入 `state-root/batch_N/depth_done.json`，格式为已完成的源 KS3 前缀字符串数组。
- `state-root/batch_N/depth_records/<hash>.json` 保存本机目录、GPU、处理状态、诊断与日志路径。
- `download-root/logs/` 保存各会话及 resident 日志。
- 重启时跳过 `depth_done.json` 中的会话。失败会话保留数据，同一次运行不反复重试；重新启动后重试。
- 仅一个深度协调器可以消费同一个 state-root。意外退出留下锁时，确认原协调器已结束后再删除 `depth_runner.lock`。

不会自动上传深度或删除输入/输出。因此共享盘使用量会持续增加；默认每个下载前要求剩余空间大于该会话大小加 50 GiB，可用 `--min-free-gb` 调整。它是预检查，不是跨进程的硬空间配额。磁盘不足的会话会记录失败，需释放空间后重启重试。

## 批量回传

`upload.py` 一次扫描当前已完成的会话（同时要求 `depth_done.json` 包含源 URI 且深度记录状态为 `done`），将整个会话目录回传到记录中的原 KS3 前缀。沿用 `ks3util -r -u`，会话依次上传，每个会话默认 64 个传输任务。失败后继续其他会话，最终以非零状态退出；不删除本地数据，也不修改深度处理记录。再次执行仍会调用工具，由 `-u` 决定哪些文件需要传输。

```bash
cd /share/project/liuyou/humancoin/code/roboego-hand-vis

/share/project/liuyou/humancoin/envs/ego/bin/python -u \
  scripts/guanglun/upload.py \
  --state-root /share/project/liuyou/humancoin/data/guanglun_batches \
  --ks3-tool /share/project/liuyou/humancoin/utils/ks3/ks3util-linux-amd64 \
  --ks3-config /share/project/liuyou/humancoin/utils/ks3/ks3util-config-team \
  --jobs 64 \
  --execute
```

去掉 `--execute` 只列出回传源目录和目标。可用 `--first-batch 1 --last-batch 1` 限定批次。该脚本处理本次扫描到的结果后退出，新增完成的数据需要再次运行。job 必须能访问记录中 `session` 指向的共享目录。

## Depth output layout

New Lightwheel jobs use shared L1 compressed HEVC12 MP4 output (CRF 0, slow,
lossless=false), with an exact FFV1 validity mask:

```text
<session>/videos/chunk-000/observation.images.depth_head_left_rectified/episode_000000.mp4
<session>/data/chunk-000/episode_000000/observation.images.depth_head_left_rectified.parquet
<session>/annotation/chunk-000/observation.images.depth_head_left_rectified/episode_000000/valid_mask_ffv1.mkv
<session>/annotation/chunk-000/observation.images.depth_head_left_rectified/episode_000000/lerobot_depth.json
```

No standalone depth directory is created. Metadata/logs remain under annotation.
Restart existing resident workers to load the changes. Existing completed datasets
are retained; this does not migrate files or reset depth_done.json.
normalize_depth.py is a legacy-layout migration utility and is not part of new jobs;
do not use it to normalize the new output layout.

## Re-encode previously normalized MKVs

Stop old normalization and depth producers before running. This command supports
the existing `depth/chunk-*/observation.images.depth_head_left_rectified/episode_*.mkv`
layout (one metadata/parquet pair per feature directory). It does not run the GPU model.

```bash
python scripts/guanglun/reencode_depth.py \
  --root /share/project/liuyou/humancoin/data/guanglun_depth \
  --first-batch 242 --last-batch 242
```

Preview is the default. Add `--execute --workers 2 --encode-threads 2` to encode.
Add `--delete-source` to remove recognized old files after output verification;
otherwise old files remain. A rerun verifies and reuses already migrated outputs.
Remove the batch limits to process all batches. Use --ffmpeg /absolute/path/ffmpeg
if FFmpeg with libx265/gray12le support is not on PATH. No ffprobe is required.

The script checks source frame count, decodes the new video and mask, verifies
the exact validity pattern and copied timestamp table, and publishes with rollback.
Legacy lossless millimeter values are decoded directly; the new HEVC encoding
uses CRF 0, slow, lossless=false. Valid depth values may change during compression.
Source timestamps were already selected by normalize_depth.py: they are copied
unchanged, indexed by depth video frame; processed_frames retains original indices.
Unknown legacy files prevent cleanup, and failed encoding leaves sources intact.
