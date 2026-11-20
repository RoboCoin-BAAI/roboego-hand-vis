# 视频人脸打码

按原有 YOLO 人脸检测与模糊流程处理视频，保留批处理的逐文件恢复机制。此阶段不作为质量门禁。

## 处理指定数据

```bash
python -m roboego_hand_vis.video_privacy_blur.run \
  --input_dir /path/to/session/videos \
  --output_dir /path/to/session/privacy/video_privacy_blur
```

此命令保持原流程，不回写数据库。

## 从 L1 数据库批量处理

完成[数据库连接配置](../database/README.md)，安装原有人脸检测依赖及项目 `database` 依赖后，在项目根目录执行：

```bash
python -m roboego_hand_vis.video_privacy_blur.database_run \
  --shared-root ./data/l1 \
  --limit 3
```

省略 `--limit` 或设为 `0` 时处理当前所有可领取 session。可用 `--config` 指定本阶段 YAML。数据库模式使用当前 Python，请直接在所需环境中运行命令。

只领取下载、头环基础质检、视频粗质检和立体矫正均为 `completed`，且 `video_privacy_blur_status=pending` 的数据。不依赖深度、手部估计或千问标注。

领取后写 `processing`；本次选择的所有视频均成功、批处理记录完整且输入/输出文件校验一致后写 `completed`；执行异常或部分视频失败写 `failed`。不自动写 `block`。支持多终端原子领取，处理期间不持有数据库事务。

每条数据从自己的 `videos` 读取，结果写入 `privacy/video_privacy_blur`，文件名为 `<原文件名去扩展名>_blurred.mp4`。原始视频保留。

默认 `parameters.videos: []` 会递归处理输入目录中的所有支持的视频，包括原始视频与矫正视频。需要限定范围时在 YAML 中设置，例如：

```yaml
parameters:
  videos: [left_rectified.mp4, right_rectified.mp4]
```

`completed` 表示本次配置选中的全部视频处理完成。设备、检测权重、模糊方法、置信度、重试次数和视频并发数仍使用 `recommended.yaml`；`device: auto` 沿用原来的设备选择。

批处理记录在输出目录的 `.batch` 下，数据库运行日志为 `database_run.log`；逐视频日志见 `.batch/logs`。失败记录不会自动重新领取，将其 `video_privacy_blur_status` 从 `failed` 改回 `pending` 后重试，批处理会复用输入、参数及输出校验一致的成功视频。

数据库模式禁止 `dry_run`，会在领取前检查权重文件和 ffmpeg/ffprobe。状态字段已经存在，不需要数据库迁移。

## 简智数据四卡原位打码

`jianzhi_inplace` 针对 `batch_0000001` 至 `batch_0000071` 的 32 位十六进制 session，处理 `videos/chunk-000` 下左右目 `episode_000000.mp4`，不处理深度视频。每张 GPU 使用一个常驻进程和一份模型，依次处理视频。成功解码并校验输出的帧数、分辨率和时间信息后，以相同路径和文件名原子替换原文件；输出不含音频，原视频不会备份。先确认这一行为符合数据发布要求。

先用 dry-run 检查范围和缺失文件：

```bash
export PATH="/share/project/liuyou/humancoin/envs/ego/bin:$PATH"
cd /share/project/liuyou/humancoin/code/roboego-hand-vis
python -m roboego_hand_vis.video_privacy_blur.jianzhi_inplace \
  --state-dir /share/project/liuyou/humancoin/data/jianzhi/privacy_blur_inplace_state \
  --start 1 --end 71 --gpus 0,1,2,3 --dry-run
```

确认后去掉 `--dry-run` 执行。指定的卡必须能被 PyTorch 看到；使用 `CUDA_VISIBLE_DEVICES` 时，`--gpus` 是该环境可见设备的编号。持久保留 state-dir：其中记录每个输入和输出的哈希，重新执行会跳过已替换成功的视频，避免重复打码；替换前后中断也可恢复。不要在运行期间修改输入、状态目录或同时用其他程序处理相同文件。检测到缺失文件时不会开始处理。每张卡的处理日志位于 state-dir 的 `logs/gpu-*.log`，结束统计位于 `summary.json`；失败的文件重运行时会重试。
