# L1 LeRobot 深度估计

读取 `videos/chunk-*/observation.images.head_{left,right}_rectified/episode_*.mp4`，
以及 `calibrations/dataset_manifest.json`。多 episode 时使用
`calibrations/chunk-000_episode_000000_dataset_manifest.json` 等带前缀的文件。
要求先运行 `lerobot_stereo_rectification.L1.run`。
视频路径由当前数据集位置重新构造，支持数据集从服务器复制到其他目录。
不检查或修正时间同步，沿用旧模型代码按同序号帧进行推理。

## 执行

无需切换工作目录，单 session 或 batch 都使用同一入口：

```bash
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_fast_foundation_depth/L1/run.py \
  --source /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001 \
  --model-root /share/project/liuyou/humancoin/code/Fast-FoundationStereo \
  --ckpt /share/project/liuyou/humancoin/code/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
  --gpus 0
```

模型路径填写实际路径。默认 `--engine throughput`，每卡一个常驻推理进程，
模型加载一次，episode 任务通过共享队列动态分配给空闲卡。
`--gpus 0,1` 表示两卡，未指定时继承 `CUDA_VISIBLE_DEVICES`，单 GPU 进程执行。
后台解码预取、GPU 推理、顺序后台保存并行，保持原 AMP dtype、
scale=0.5 和 valid_iters=8；高吞吐模式会拒绝改变这两个质量参数的配置。
支持按 `--batch-size 4` 将 4 对连续左右目帧作为一个 batch 输入模型；默认 1，
末尾不足一个 batch 的帧也会正常处理。增大 batch 可能显著增加 GPU 显存占用，
请从 2 开始试，并按 GPU 显存调整。此参数仅适用于 `--engine throughput`。
预览默认关闭。
日志在 `<session>/annotation/depth_chunk-000_episode_000000.log` 等逐 episode 文件中。

性能参数可加 `--batch-size 2 --prefetch 4 --write-queue 4 --cpu-threads 2`：
分别控制每次模型调用的帧对数、每卡预取帧数、待写帧数和 CPU/解码线程数。队列有界，
不会把整段视频载入内存。先用默认值，再根据统计调整。
`--engine legacy` 可退回之前每个 session 启动模型的串行帧处理实现。

固定帧偏移对照试验：`--right-frame-offset -1` 表示左目第 i 帧配右目第 i−1 帧，
0 表示原同序号配对，+1 表示配右目第 i+1 帧。不补帧，只裁去无对应帧的边界。
偏移非零时，输出自动改为 `depth/chunk-000/episode_000000_right_offset_m1/`
或 `..._right_offset_p1/` 等独立目录，日志也单独命名；重复同偏移试验覆盖其试验结果，
不会覆盖无偏移基线。帧号和 source_timestamps.parquet 仍以左目原始帧号为准，
右目映射写入 lerobot_depth.json。此功能仅适用于 throughput，且不是全局时间同步修复。

默认 `--num-frames auto --start 0`，处理左视频全部帧，不沿用旧推荐配置的 600 帧限制。
可用 `--start 0 --num-frames 100` 先试跑。参数 `scale`、迭代次数、
最大深度、PNG/NPY/预览开关通过 `--config /path/to/config.yaml` 设置，
格式与旧 `fast_foundation_depth/recommended.yaml` 相同。
默认使用本目录独立的 `recommended.yaml`：scale=0.5、valid_iters=8、max_depth_m=4，
保存 PNG，不保存 NPY，预览默认关闭。为保证旧顺序读取器正确前进，
高吞吐模式每次重算请求范围到临时目录，全部推理和写盘成功后替换该 episode
深度目录；处理失败保留之前的输出。指定部分帧时，发布的目录仅包含该范围，
不会混入旧范围的结果。legacy 模式覆盖指定范围但保留范围外旧文件。

`--dry-run` 仅解析输入、列出任务和输出位置，不加载模型、不要求 CUDA，不写文件。
仅显式传入命令行 `--save-vis` 才开启预览，默认关闭并覆盖 YAML 中的预览开关；
`--preview-every-n-frames 5` 设置预览间隔，命令行设置优先于 YAML。
高吞吐预览默认使用 `--vis-min-depth 0.2 --vis-max-depth 1 --vis-mode linear`，
将 0.2–1 米铺满 TURBO 色域，近处暖色、远处冷色，超出范围截到对应端点颜色，
无效深度为黑色。只改变显示，不改变 4 米有效深度范围或存储值。
支持显式选择 inverse 模式，但此时 vis-min-depth 必须大于 0。

## 环境

需要 CUDA GPU、兼容的 PyTorch，以及 Fast-FoundationStereo 源码与其依赖。
模型需为旧实现支持的序列化 `.pth`，旁边必须有 `cfg.yaml`，普通 state_dict 权重不能直接替代。
Python 依赖沿用旧推理环境，包括 numpy、opencv-python、PyYAML、omegaconf
及模型仓库本身的依赖；输入发现不需要安装 LeRobot。

## 输出

```text
<session>/depth/chunk-000/episode_000000/
├── depth_uint16_png/frame_000000_depth_mm.png
├── vis/depth_preview.mp4
├── source_timestamps.parquet
├── lerobot_depth.json
└── fast_foundation_stereo_video_meta.json
```

PNG 为 uint16 毫米深度，0 无效；开启 `save_npy` 后另存米制 float32 NPY，NaN 无效。
文件编号保留原矫正视频帧号，深度位于矫正左目坐标系。
时间戳侧文件为完整左目时间戳，按帧号查行；实际处理范围见 `lerobot_depth.json`。
高吞吐报告还包含 FPS、总耗时、累计解码耗时、CUDA Event 模型耗时、
推理与传输耗时及保存耗时。各阶段重叠执行，累计耗时不应简单相加。
真实 GPU 吞吐及数值一致性需在对应模型和 CUDA 环境验证，不能由 CPU 测试推断。
目前深度以单独 PNG/NPY 模态附件保存，不冒充普通 RGB 视频特征，
不修改原 `meta/info.json`、视频或标定。旧模型缺乏 CUDA 时会报错，未做 CPU 回退。

## 直接保存 HEVC 12-bit，避免逐帧文件

高吞吐模式可在原命令中增加：

```bash
--depth-format hevc12 \
--ffmpeg /share/project/liuyou/humancoin/envs/ego/bin/ffmpeg \
--hevc12-preset ultrafast \
--encode-threads 2
```

深度在后台写入线程直接送给 FFmpeg，不落地 PNG/NPY，输出一个
`depth_hevc12.mkv`。此模式覆盖配置中的 save_depth_png/save_npy，
预览仅在显式传入 `--save-vis` 时开启。要求 FFmpeg 的 libx265 支持 gray12le。
采用真正的 `lossless=1`、gray12le 输入输出，不经过 gray16 色阶缩放，
0 无效、1 至 4095 表示毫米，故 max_depth_m 不得超过 4.095（默认 4.0）。
使用 scale=0.5、valid_iters=8；与现有 PNG 的毫米量化一致，
不是对浮点 NPY 深度的无损保存。此模式不需要额外有效性 mask，0 已可精确保留。

每个 episode 只保存一个深度 MKV、时间戳、元数据、编码日志和可选预览。
`lerobot_depth.json` 的 depth_storage 包含单位、无效值、帧数、尺寸、FPS 及索引映射。
读取使用 `depth_video.read_depth_video(path, size=(width, height), ffmpeg=...)`，
解码为 float32 米深度，0 恢复为 NaN；不要使用普通 RGB 视频读取器读取毫米值。

libx265 使用 CPU 编码，可能成为高吞吐流水线的新瓶颈。preset 影响压缩率和速度，
不改变 lossless=1 的毫米整数精度；默认 ultrafast 优先编码速度，文件可能更大，
再按 CPU 资源调整线程数。
旧归档模块及其默认配置不变；其 hevc12_lossless=false 与本模式的显式无损不同。
