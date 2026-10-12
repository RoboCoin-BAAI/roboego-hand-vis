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
