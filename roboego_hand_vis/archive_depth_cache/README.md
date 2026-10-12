# 深度缓存归档

把逐帧深度缓存归档为视频。此步骤必须在手部估计完成后运行。

## 处理指定数据

```bash
python -m roboego_hand_vis.archive_depth_cache.run \
  --depth_dir /path/to/session/depth
```

此命令保留原流程，不回写数据库。

## 从 L1 数据库批量处理

完成[数据库连接配置](../database/README.md)后，在具备项目 `database` 依赖、ffmpeg 和 ffprobe 的环境中执行：

```bash
python -m roboego_hand_vis.archive_depth_cache.database_run \
  --shared-root ./data/l1 \
  --limit 3
```

省略 `--limit` 或设为 `0` 时处理当前所有可领取数据。使用 `--config` 指定其他归档 YAML，默认使用本阶段 `recommended.yaml`。不需要设置 CUDA 设备。

只领取下载、前两项质检、立体矫正、深度前门禁、深度估计及手部估计均为 `completed`，且 `archive_depth_cache_status=pending` 的数据。这里要求的是 `hand_status`，不是可选手部采样检测的 `detect_hand_status`。

领取后写 `processing`；配置要求的归档视频均存在、实际解码帧数与源深度帧数一致后写 `completed`；执行异常、缺少输出或帧数不符写 `failed`。本阶段不作为质量门禁，不自动写 `block`。多终端原子领取，编码期间不持有数据库事务。

每条数据使用自己的 `depth` 目录，归档输出仍放在此目录。默认输出包括：

- `depth_hevc12.mkv`：12-bit HEVC 深度视频。
- `valid_mask_ffv1.mkv`：有效深度 mask。
- `depth_uint16_ffv1.avi`：16-bit FFV1 深度视频。
- `depth_preview.mp4`：深度预览。
- `depth_archive_meta.json`：归档元数据。

处理日志在 `annotation/archive_depth_cache.log`。数据库模式不允许 `dry_run_avi`，并要求至少启用一种深度视频输出（`write_avi` 或 `write_hevc12`）。

默认配置 `delete_png: true`，会删除逐帧 PNG；`delete_npy: false`，保留 NPY。数据库入口先保留源缓存，完成所有视频验证后再按配置清理，验证失败保留源缓存。若希望始终保留 PNG，将 `delete_png` 改为 `false`。如果 NPY 与 PNG 混合缓存中存在未被 NPY 覆盖的 PNG 帧，会在编码前报错并保留缓存。清理只删除已归档的帧。源缓存已删除时不会自动重新归档。

状态字段已经存在，不需要新增表字段或迁移数据库。若手部估计尚未接入数据库，需要确认其完成结果并将 `hand_status` 记为 `completed`，归档入口才会领取。
