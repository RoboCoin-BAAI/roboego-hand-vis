# 简智 1–71 双目与深度整理

脚本：`prepare_jianzhi_stereo_release.py`，仅使用 Python 标准库。

按确认的范围执行，不检查、解码或重编码保留的视频，也不读取 Parquet 内容。

- 保留去畸变左目、右目和深度的视频目录及对应 Parquet。
- 其他视角的视频目录、`observation.images.*.parquet` 移到发布目录外，保留原相对路径。
- 修改 `meta/info.json`：删除其他视频特征，主流设为 `observation.images.head_left_rectified`，`codebase_version` 从 `v3.1` 改为 `v2.1`，`total_videos` 按每个 episode 三路设置。
- 原 `info.json` 先备份；单个样本执行失败时尝试恢复该样本。
- 不改 `readme.json`、统计文件、标注、主时间索引或非视频 Parquet。README 由用户另行重新生成。

```bash
cd /share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools
/share/project/liuyou/humancoin/envs/ego/bin/python prepare_jianzhi_stereo_release.py \
  --root /share/project/liuyou/humancoin/data/jianzhi/wait_for_publish \
  --start 1 --end 71 \
  --backup-root /share/project/liuyou/humancoin/data/jianzhi/stereo_depth_backup_20261118 \
  --report /share/project/liuyou/humancoin/data/jianzhi/stereo_depth_apply_20261118.jsonl \
  --workers 16 --apply
```

省略 `--apply` 仅列出待整理的数量。重复执行已完成的样本不会再次修改；尚未完成且备份已存在的样本会报错，避免覆盖原备份。报告应使用新的文件名，避免覆盖之前的执行记录。

## 2026-11-18 执行结果

- 简智 1–71 共 7,078 个样本，全部完成 `info.json` 更新。
- 共 42,468 个旧视角 Parquet 移入外部备份，即每个样本六个；本次没有其他视频需要移走。
- 原始 `info.json` 同时备份。`readme.json`、统计文件、标注和其他 Parquet 未修改。
- 并行任务的重复尝试被备份目录的独占创建机制阻止；汇总后未完成样本为零。
- 所有样本写入后均已逐一核对：再次生成修改计划为空。额外启动的重复只读扫描已停止，不再重复遍历共享存储。
- 汇总：[jianzhi_stereo_depth_summary_20261118.json](jianzhi_stereo_depth_summary_20261118.json)。服务器汇总：`/share/project/liuyou/humancoin/data/jianzhi/stereo_depth_summary_20261118.json`。
