# 深度估计

使用 Fast-FoundationStereo 和模型权重在 CUDA 环境下生成逐帧深度，依赖和模型路径仍按 `recommended.yaml` 配置。

## 处理指定数据

```bash
CUDA_VISIBLE_DEVICES=0 python -m roboego_hand_vis.fast_foundation_depth.run \
  --manifest /path/to/session/dataset_manifest/dataset_manifest.json \
  --output_dir /path/to/session/depth
```

## 从 L1 数据库批量处理

完成[数据库连接配置](../database/README.md)，先运行[深度前质量门禁](../stereo_depth_gate/README.md)，再在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m roboego_hand_vis.fast_foundation_depth.database_run \
  --shared-root ./data/l1 \
  --limit 3
```

运行环境需要安装项目的 `database` 依赖。可用 `--config` 指定深度配置；GPU 由 `CUDA_VISIBLE_DEVICES` 指定，子进程继承该设置。省略 `--limit` 或设为 `0` 时处理当前所有可领取数据。

只领取下载、前两项质检、立体矫正和深度前门禁均已 `completed`，且 `fast_foundation_depth_status=pending` 的数据。领取后写 `processing`，配置范围内的深度输出齐全后写 `completed`；执行异常或缺帧写 `failed`，此阶段不作为门禁。

每条数据使用自己的 `dataset_manifest/dataset_manifest.json`，输出至 `depth`，日志为 `annotation/fast_foundation_depth.log`。多终端领取不会重复处理同一条数据，推理期间不占用数据库事务。

当前默认配置 `num_frames: 600` 只处理 600 帧；需要整个视频时，在深度配置的 `parameters` 中改为 `num_frames: auto`。`completed` 表示配置范围内处理完成。不会自动启动手部估计或深度归档。
