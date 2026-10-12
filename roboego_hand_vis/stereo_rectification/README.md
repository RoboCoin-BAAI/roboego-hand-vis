# 去畸变与立体矫正

当前仅支持标定中的 `camera_model: pinhole` + `distortion_model: equidistant`。双球面实现尚未完成，检测到后报错停止；左右目模型和参数数量在处理前校验。

## 处理指定数据

```bash
python -m roboego_hand_vis.stereo_rectification.run \
  --session /path/to/session
```

输出 `videos/left_rectified.mp4`、`videos/right_rectified.mp4` 和 `dataset_manifest/dataset_manifest.json`。参数见 `recommended.yaml`，此命令不回写数据库。

## 从 L1 数据库批量处理

完成[数据库初始化与连接配置](../database/README.md)后执行：

```bash
python -m roboego_hand_vis.stereo_rectification.database_run \
  --shared-root ./data/l1 \
  --limit 3
```

`--shared-root` 使用下载时的共享根目录。省略 `--limit` 或设为 `0` 时处理所有当前可领取的数据。支持 `--config`、`--balance`、`--width`、`--height`、`--overwrite` 等原处理参数；数据库模式不允许 `dry_run`。

只领取下载、基础质检和视频粗质检均为 `completed`，且立体矫正为 `pending` 的 L1 数据。手部采样检测是可选阶段，无需等待它完成。

运行期间 `stereo_rectification_status=processing`，原处理命令成功且矫正视频、manifest 均存在并非空时写 `completed`；不支持的模型或执行异常写 `failed`。此阶段不作为门禁，不自动写入 `block` 或启动后续阶段。

每条数据的执行日志写入 `annotation/stereo_rectification.log`。批次有失败时退出码为 `1`。
