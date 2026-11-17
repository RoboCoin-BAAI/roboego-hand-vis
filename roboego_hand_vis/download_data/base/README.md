# 共用 KS3 批次下载

在仓库根目录执行（PowerShell）：

```powershell
python -m roboego_hand_vis.download_data.base.run `
  --ks3-tool C:\path\ks3util.exe `
  --ks3-config C:\path\ks3util-config `
  --batch-csv C:\data\batch_0000001.csv `
  --output-dir D:\data\batch_0000001 `
  --jobs 32
```

也可直接运行 `python roboego_hand_vis/download_data/base/run.py`，参数相同。

CSV 使用 UTF-8（支持 BOM），单列表头为 `目标路径`，每行一个 `ks3://bucket/path/session/` 目录。不限制数据集名称，重复的末级目录名会在下载前报错。

每条数据递归下载到 `output-dir/<session>/`。CSV 中各条顺序处理，`--jobs` 控制 ks3util 内部传输并发，默认 64，必须为正整数。使用 ks3util 的 `-u` 更新模式，重复执行时由工具判断需要传输的文件。

单条下载失败后继续处理后续数据，最后汇总成功、失败数量；有失败则退出码为 1。失败时可能保留部分文件，可重新执行同一命令。

旧入口 `roboego_hand_vis.download_data.L1.run` 保留并转调此实现，支持相同参数。
