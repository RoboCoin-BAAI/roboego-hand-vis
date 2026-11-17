# 通用单目深度批处理

`run_base_mono_depth.py` 负责发现任务、多 GPU 常驻进程、验证和原位发布。
实际推理调用 `RoboCOIN-with-Depth/src/depth_repair/da3.py` 的
`create_depth_repair('da3').repair_frames(...)`；编码复用该项目的
`DepthVideoEncoder`，不调用其数据集增强器，以免写入额外文件。

## 光轮运行

环境沿用 DA3 的 `numpy<2` 要求。时间戳处理额外依赖 PyArrow；在 NumPy
1.26.4 环境中使用 `pyarrow==17.0.0`，不要为 PyArrow 升级 NumPy：

```bash
/share/project/liuyou/humancoin/envs/robocoin-depth/bin/python -m pip install \
  --no-deps --only-binary=:all: pyarrow==17.0.0
```

正式执行前脚本会检查依赖可导入，再启动 GPU 模型加载。

先预览，不导入模型，不创建 work-dir，也不修改数据：

```bash
/share/project/liuyou/humancoin/envs/robocoin-depth/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/scripts/run_base_mono_depth.py \
  --root /share/project/liuyou/humancoin/data/guanglun_depth \
  --work-dir /share/project/liuyou/humancoin/mono_depth_work/guanglun \
  --gpus 0,1,2,3,4,5,6,7 \
  --dry-run
```

第一次只覆盖一个视频做验证：

```bash
export PATH="/share/project/liuyou/humancoin/envs/robocoin-depth/bin:$PATH"
/share/project/liuyou/humancoin/envs/robocoin-depth/bin/python -u \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/scripts/run_base_mono_depth.py \
  --root /share/project/liuyou/humancoin/data/guanglun_depth \
  --work-dir /share/project/liuyou/humancoin/mono_depth_work/guanglun \
  --robocoin-root /share/project/liuyou/humancoin/code/RoboCOIN-with-Depth \
  --gpus 0 --limit-videos 1
```

批量运行时去掉 `--limit-videos 1`，将 `--gpus` 改成所需 GPU 列表。
可用 `--first-batch 1 --last-batch 10`、`--session <会话名>` 限制范围。
默认匹配 `batch_*/*`；其他目录结构用 `--session-glob`，例如 `'*'`。
默认相机通过 `--camera` 指定，目标字段通过 `--depth-feature` 指定。
相同 work-dir 和参数重启会跳过已成功发布且输入、输出未变化的视频；
`--force` 会重新推理并覆盖。已经存在的双目深度不算作本脚本完成记录。

## 写入范围

原始 session 只写入或覆盖：

```text
videos/chunk-*/observation.images.depth_head_left_rectified/episode_*.mp4
data/chunk-*/episode_*/observation.images.depth_head_left_rectified.parquet
meta/info.json
```

时间戳 Parquet 直接复制对应左目 RGB 的表，保留所有列。验证 RGB 解码帧数、
模型输出帧数和时间戳行数一致，时间戳有限且递增，输出完整可解码。
info.json 保留其他字段，替换深度字段的旧编码信息，并更新 total_videos。
不修改已有 annotation、episodes_stats.jsonl 或原始 RGB。若旧 annotation 仍在，
其中的双目有效性掩码不适用于本次单目结果；清理按用户独立安排执行。

日志 `gpu-*.log`、任务完成记录 `tasks/*/done.json` 和所有临时文件都在 work-dir。
发布前将原文件临时备份到 work-dir，成功后删除备份；异常回滚，进程中断后
下次使用相同 work-dir 启动会恢复未完成事务。三项文件分别原子替换，
不保证并发读取者看到三项同时切换；运行时不要有其他程序写同一批数据。
work-dir 必须位于数据目录之外，且与数据同一文件系统，以支持原子替换。
根目录互斥锁位于系统临时目录；不要对重叠的不同 root 同时启动任务。

## 深度编码及资源

DA3 代码将预测乘 1000，按毫米返回；这代表该实现的数值约定，不代表实测精度。
沿用原项目未开启 normalize 的编码：不缩放、不四舍五入，直接将毫米数组交给
原始 DepthVideoEncoder，执行 clip(0, 4095) 后转 uint16。
像素存储值即毫米值，超过 4095 mm 截断为 4095，负值截断为 0。
单位、模型、推理及编码参数均写入深度字段的 info；不做逐视频归一化。
默认 HEVC gray12le，按用户要求使用 CRF 0、preset fast；数值编码方式沿用原项目。
不再额外限制 x265 的 pools 或 frame-threads；仅对小数帧率修正整数 GOP 参数。
旧版 16 mm/单位的完成记录不会复用；更新后需重新生成对应深度。

每张指定 GPU 一个进程，启动加载一次模型，按 session 动态分配任务；
同一 session 的视频顺序执行，避免 info.json 并发覆盖。卡上模型跨任务常驻，
所有任务完成后退出，不持续轮询新增数据。单个视频失败会继续其他视频，
模型加载失败或进程崩溃则整个命令报错退出，不静默漏任务。
默认分辨率 504，按用户要求使用 chunk-size 80、overlap 20。
RGB 按 chunk 解码并后台预取 1 个块；深度按原版低分辨率 scale/shift
拟合及融合规则衔接，再逐帧恢复原分辨率、截断并写入原版 FFmpeg 编码命令。
使用独立的有界队列让解码、GPU 推理和编码并行，深度队列最多缓存 2 个块。
队列之外仍有正在解码、推理、编码的块和重叠帧，因此实际内存不止两个块。
队列满时生产者等待，不保存整段 RGB、整段全分辨率深度或整段编码数组，CPU 图像内存主要随
chunk-size 和并行卡数增长，不再随视频总帧数增长。仍需为模型、解码和编码预留内存。
编码较慢且缓冲填满时 GPU 仍可能等待；不代表模型被卸载。
校验先读取视频头，再通过一次 FFmpeg 完整解码同时检查帧数与解码错误，
不再先 ffprobe 完整计数后再重复解码。
实际网络 forward 及预处理仍调用 RoboCOIN-with-Depth 中的 DA3；原版 simple 路径
继续复用 repair_frames。重叠路径仅调整数组生命周期，未修改拟合、融合和缩放公式。
GPU 日志新增每块帧区间和耗时（包含编码背压），FFmpeg 错误也写到同一日志。
修复工作进程异常退出后任务队列 feeder 导致主进程退出挂起的问题，清理等待有上限。
成功记录的 adapter_version 保持为 2，相同推理和编码参数下继续复用已有成功记录。
不生成抽帧图片、逐帧 NumPy 或可视化视频。

## 本地检查

```powershell
python -m unittest discover -s tests -p test_base_mono_depth.py -v
```

这些 CPU 测试覆盖筛选、回滚、恢复、元数据、无写入预览、解码缓冲上限及
与原版重叠对齐/缩放公式的数值一致性；实际 GPU 推理和
最终数据兼容性仍需要上述单视频试跑验证。
