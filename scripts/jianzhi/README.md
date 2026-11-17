# Jianzhi pipeline

## 逐批下载并计算深度（模型常驻）

批次按编号逐个查询，默认从 1 到 200（含），不再依赖启动时列出的根目录快照。
用 --first-batch 22 --last-batch 200 可接续处理；每批在轮到时查询数据 ID。
不存在、为空或查询失败的批次记录在外部 failed/<batch>/discovery.json，继续后续编号。
跳过的批次本轮不会自动等待或回头重试，后续可指定范围重跑。

简智入口首次深度 HEVC12 编码默认使用 medium（depth_batches.py、depth.py）。
depth_batches.py 和 depth.py 本地模式可用 --hevc12-preset 覆盖。
已有通过验证的批次完成结果继续复用，不因预设调整重算或重编码。

默认不需要提供 CSV 或 --batch-dir，直接枚举 --remote-root 下规范的 batch/ID/。
每次进入一个批次才枚举该批 ID；自动清单保存在 runtime-dir/inventory，
不写入数据目录。--batch-dir 仅保留为旧的可选模式。

`depth_batches.py --batch-dir <CSV目录> --source <本地根目录>
--runtime-dir <外部运行目录> --ks3-tool <工具> --ks3-config <配置>
--model-root <FFS源码> --ckpt <权重> --gpus 0,1 --batch-size 4 --valid-iters 4`
从 wait_for_publish/batch/ID/ 下载整批所需输入，整批计算成功后再下载下一批。
各 GPU 的模型首次执行任务时加载，跨批次常驻。默认处理所有帧、输出 HEVC12。
仅下载矫正双目视频和所有非视频目录，不上传、不删除本地数据。
完成记录和日志在 runtime-dir；同参数且结果校验有效的成功项可重跑复用。
单条下载、推理或完成校验失败仅记入 runtime-dir/failed，继续成功项及下一批。
临时编码工作目录位于 runtime-dir/work，发布临时目录也位于数据集之外；不发布 FFmpeg 日志。
--dry-run 查询下载计划但不运行 GPU。
可用 --first-batch/--last-batch 限制批次。不要在同一数据上并行启动多个写入流程。

## 仅下载深度所需数据

`download_depth.py` 按批次 CSV 查询远端文件清单，只下载 videos 下的
`observation.images.head_left_rectified` 和 `observation.images.head_right_rectified`
MP4，其他非 videos 目录全部下载。不会删除已经下载的其他相机。
缺少完整矫正双目或 meta/info.json 时，该条报错并继续下一条。
输出保持 `<output-dir>/<batch>/<ID>/`，runtime-dir 必须在 output-dir 外。

```bash
python scripts/jianzhi/download_depth.py \
  --batch-dir /path/to/batches_2026-11-05 \
  --output-dir /home/jianzhi \
  --runtime-dir /share/project/liuyou/humancoin/jianzhi_depth_download_runtime \
  --ks3-tool /share/project/liuyou/humancoin/utils/ks3/ks3util-linux-amd64 \
  --ks3-config /share/project/liuyou/humancoin/utils/ks3/ks3util-config-team \
  --jobs 64 --dry-run
```

去掉 --dry-run 开始下载。也可用 --batch-csv 只跑一个批次。
CSV 仅用于确定批次和 ID，不使用其中的原始下载地址。默认从
`ks3://baai-eai-pwm/jianzhi/wait_for_publish/<batch>/<ID>/` 查询，
可通过 --remote-root 覆盖此已处理数据前缀。
这不是原始数据首次下载入口，不能用于尚未矫正的数据。
depth.py 的上传收据下载模式同样只下载两路矫正视频及所有非视频文件。

## 深度估计

`depth.py` 参考光轮深度脚本，复用 `lerobot_fast_foundation_depth.base`。
简智相机名与 L1 一致，本地模式通过 L1 CLI 接入 base；已有矫正数据无需再次去畸变。

先验证本地数据（不加载模型、不写深度）：

```powershell
python scripts/jianzhi/depth.py --source D:\data\jianzhi\001301e26b8c4ace844c7b2d12eacf9d --dry-run
```

正式计算需要 CUDA、Fast-FoundationStereo 源码、权重及权重旁的 `cfg.yaml`：

```bash
python scripts/jianzhi/depth.py --source /path/to/session \
  --model-root /path/to/Fast-FoundationStereo \
  --ckpt /path/to/model_best_bp2_serialize.pth \
  --engine throughput --gpus 0 --batch-size 1 --depth-format hevc12
```

`--source` 支持单条或批次目录，其余参数沿用 L1 CLI，可用 `--source /path --help` 查看。
默认发布深度视频到 `videos/chunk-*/observation.images.depth_head_left_rectified/episode_*.mp4`，
掩码及深度元数据在 `annotation/chunk-*/observation.images.depth_head_left_rectified/episode_*/`，
时间戳保存在 data 下对应深度特征中，并更新 `meta/info.json`。
manifest 的旧机器路径会按当前 session 重建；双目按同帧号读取，不自动检查时间同步。

也支持与光轮相同的上传后下载、常驻 GPU worker 模式：

```bash
python scripts/jianzhi/depth.py \
  --state-root /share/project/liuyou/humancoin/data/jianzhi_pipeline_state \
  --download-root /home/jianzhi_depth \
  --model-root /path/to/Fast-FoundationStereo \
  --ckpt /path/to/model_best_bp2_serialize.pth \
  --ks3-tool /path/to/ks3util --ks3-config /path/to/ks3-config \
  --gpus 0,1 --depth-workers-per-gpu 1 --batch-size 1 --once --execute
```

此模式读取 `batch_*/records/*.json` 的 `uploaded/complete` 记录及上传文件收据，
跳过有 error 的记录和 `depth_done.json` 中已完成的 URI；兼容上游已删除本地数据的 complete 状态。
省略 `--execute` 只打印待处理数量；省略 `--once` 则持续轮询。
计算结果留在 download-root，不自动上传或删除。完成状态写入 state-root 下的
`depth_records/` 和 `depth_done.json`，日志在 download-root/logs。
本地模式与上传收据模式的参数分别使用，不混合传入。

结构化标注完成后自动调用 `build_json.base.build_readme` 生成正式 `meta/readme.json`，成功后才进入上传。日志在外部 state-root 对应条目的 build_readme.log；生成失败停止该条后续上传并保留数据。单条入口同样生效。旧 annotated 状态恢复上传时，如尚未记录生成成功，会补生成 readme；不重做去畸变或标注。

数据目录不存放锁、模型请求收据、ffmpeg 日志或去畸变临时目录。简智去畸变工作目录位于系统临时目录 `jianzhi-rectification/`，标注运行状态位于 `argus-jianzhi-text/`；上传前会将旧版本遗留的已知运行文件迁出到 state-root 的 logs/ID/runtime_artifacts。正式校正视频、时间戳、校正参数和最终标注保留。

单条完整流程测试入口：[single.py](single.py)，参数与 Windows 示例见 [SINGLE.md](SINGLE.md)。单条入口复用本流水线的处理和状态逻辑，不需手动建 CSV。

## 标注阶段更新：复用原始文本

当前 pipeline 标注阶段调用 Argus `review.jianzhi.structure_existing`，读取 `meta/annotations.json`，不再对视频重新标注、不抽帧。去畸变仍作为独立数据处理阶段保留。原始叶子动作段、时间边界、描述及成功标记保留；默认 qwen3.7-plus 仅根据原文及 SST 补任务名、动作规范标签、明确提及的物体和功能场景。不凭空生成手别、可见性、进度或视觉属性，缺失值为 null/空数组。原始 annotations.json 不变，结果在 annotation/episode_*_structured.json。

默认仍为整批下载 → 8 进程去畸变 → 8 进程文本结构化 → 上传 → 校验后删除。移除 sample-fps 和 annotation-budget-cny 参数。以下旧章节的视频采样描述已由本节替代。此前已标记 annotated/uploaded/complete 的状态不会自动重算，应使用新的 state-root 启动新处理；已有不同来源的最终标注会拒绝复用，需要明确处理或 --force 重算。

## 完整 batch 流程

`pipeline.py` 按 CSV 顺序逐批处理，batch 成员和编号保持不变：下载到
`local-root/<batch>/<ID>/` → 双目去畸变 → 基于去畸变左目的结构化语义标注 →
上传 `ks3://baai-eai-pwm/jianzhi/wait_for_publish/<batch>/<ID>/` → 校验远端清单后删除本地该条数据。
上传整个数据集，包括原始及去畸变视频、parquet、标定、meta 和最终 annotation。
不按场景或任务名称更改上传路径。

```bash
python scripts/jianzhi/pipeline.py \
  --batch-dir /share/project/liuyou/humancoin/data/jianzhi_batches \
  --local-root /home/jianzhi \
  --state-root /share/project/liuyou/humancoin/data/jianzhi_pipeline_state \
  --argus-repo /share/project/liuyou/humancoin/code/argus \
  --ks3-tool /share/project/liuyou/humancoin/utils/ks3/ks3util-linux-amd64 \
  --ks3-config /share/project/liuyou/humancoin/utils/ks3/ks3util-config-team \
  --ffmpeg /share/project/liuyou/humancoin/envs/ego/bin/ffmpeg
```

使用安装了去畸变及 Argus 依赖的 Python 环境运行；模型配置沿用 Argus 的
`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`。默认模型按用户指定的 `qwen3.7-plus`
原样传入，视频采样 1 FPS，结构化提取也沿用该模型。可用 `--model`、`--sample-fps` 覆盖。
模型 ID 是否在所配服务端可用需由服务端确认，脚本不自动替换。
该模型没有本地价格估算，默认不传仅适用于 qwen3.8-27b 的 CNY 预算参数。

以上服务器路径均已设为默认值，在仓库根目录直接运行：

```bash
/share/project/liuyou/humancoin/envs/ego/bin/python scripts/jianzhi/pipeline.py
```

每批严格依次执行，阶段间不重叠：

1. 复用 `download_data.base.run`，逐条下载整批数据到 `/home/jianzhi/<batch>/<ID>/`。不建立下载进程池、不传 jobs 参数，沿用 base 的 KS3 内部并发默认值 64。每条的下载 CSV 保存在 state-root 的对应日志目录中，便于断点续跑。
2. 整批下载结束后，8 个进程只做双目去畸变和立体校正。
3. 整批去畸变结束后，8 个进程对成功项执行标注，每条模型请求并发 1；标注复用 Argus base，默认 qwen3.7-plus、1 FPS，通过 Schema 校验并保存后清理标注临时工作目录。
4. 整批标注结束后，逐条上传去畸变和结构化标注成功项，不额外开上传进程池；传输并发沿用 KS3 jobs=64。
5. 上传阶段结束后，仅对上传成功项再次校验远端文件并删除本地数据，然后处理下一批。失败项的本地目录保留。

参数为 `--rectify-workers 8 --annotation-workers 8`，均已设为默认值。不再提供 download-workers、prepare-workers 或 upload-workers，下载和上传仅使用 KS3 内部并发。
单条数据在任一阶段失败，记录到 state-root 下该批的 `skipped.json`，从后续阶段排除，不阻塞其他数据或下一批。重新运行也跳过这些失败项，不自动重试。所有批次处理完若有失败项，退出码为 1，仅用于汇总提示。
只跑一批使用 `--batch-csv /path/batch_0000001.csv` 替代 `--batch-dir`。
加 `--dry-run` 只打印路径计划，不下载、不调用模型、不上传或删除。

状态和阶段日志保存在独立 state-root 中，不能与 local-root 重叠。成功阶段写入状态后
才进入下一阶段；上传收据持久化后，再次列举远端核对所有文件名和大小，本地清单未改变
才允许删除。上传使用 ks3util 的传输校验；远端复查是文件大小校验，不是逐文件下载验哈希。
任一步失败保留本地数据和日志，后续运行跳过；正常中断而没有失败记录的数据可从已完成阶段继续。断电后若有 `pipeline.lock`，
确认旧进程已停止后再移除锁文件。不接管已有但未登记在状态中的本地数据目录。
最终结构化 JSON 放在数据集 annotation 根目录，Argus 工作目录在系统临时目录。

