# 光轮本机批次流程

仅运行下列命令时才访问 KS3。先在计算节点生成清单，再处理；默认每批 100 条、16 个工作进程，每个进程依次下载、转换、上传。一个批次全部结束才开始下一批。视频和 MCAP 只落到本机 `/home`。

```bash
python -m roboego_hand_vis.download_data.guanglun.run inventory \
  --state-root /share/guanglun_batches \
  --ks3-tool /path/to/ks3util --ks3-config /path/to/ks3.conf

python -m roboego_hand_vis.download_data.guanglun.run run \
  --state-root /share/guanglun_batches --local-root /home/guanglun_work \
  --converter-repo /path/to/Multi-Frequency-LeRobot \
  --workers 16 --ks3-tool /path/to/ks3util --ks3-config /path/to/ks3.conf
```

转换环境需安装 Multi-Frequency-LeRobot 的依赖和 ffmpeg，可用 `--python`、`--ffmpeg` 指定。上传目标默认 `ks3://baai-eai-pwm/LightwheelAI/EgoStandard_lerobot/batch_0000001/<session>/`。

共享盘记录：

- `inventory.json`：完整源 URI、大小、ETag、固定批次、上传目标；已有清单禁止变更批次编号。
- `batch_0000001.csv`：与 L1 相同的单列路径 CSV，每批最多 100 条。
- `batch_0000001/manifest.json`：本批完整对象清单。
- `batch_0000001/records/<id>.json`：每条下载、转换、失败、上传核验收据和本机清理状态；上传收据包括所有相对文件名及大小。
- `batch_0000001/progress.json`：本批记录汇总。
- `batch_0000001/rectification_done.json`：已转换且上传核验成功的 **KS3 数据集前缀字符串数组**，供后续深度流程下载使用。源视频为 `native_rectified`，不是本流程重新去畸变。

深度端读取每个前缀，递归下载到本机，再把本机数据集目录传给深度程序。现有 L1 GPU 脚本要求本地目录，不能直接把这个远端索引作为输入。

上传逐文件执行，远端列举核对所有文件大小后才持久化 `uploaded`，随后删除对应的整个本机工作目录（包含 MCAP、转换结果与临时文件）。核验失败保留本机数据；失败批次阻止继续下一批，避免磁盘累积。修复原因后重复运行可恢复，已上传的条目仅重试清理。姿态质检退出码 2 记录为 `rejected`，不上传，诊断保存在对应记录中。

可加 `--batches batch_0000001` 选择批次。只允许一个协调进程操作同一个状态目录；意外退出留下 `runner.lock` 时，确认原进程已结束后再删除锁。中断后每条记录为恢复依据，下一次运行会重建汇总。

????????????????? `conversion_failed`?????????????????????????????????? `Invalid semantic interval` ?? `failed` ????????????????????????????????????
