# 简智数据批次

Pipeline: [scripts/jianzhi/README.md](../../../scripts/jianzhi/README.md)

`inventory.py` 从 `ks3://baai-eai-pwm/jianzhi/` 逐层列举目录，遇到 32 位十六进制 ID 即视为一条数据，不继续进入其 data/meta/videos 等子目录。根目录下 `_` 开头的日志、状态目录会跳过。扫描失败时不会输出不完整的批次。

默认按完整路径排序，每批 100 条，文件名为 `batch_0000001.csv`，单列表头为 `目标路径`，UTF-8 BOM 编码，与 L3 参考批次格式一致。`inventory.json` 保存完整路径、批次数量和排除目录。输出目录已有清单时拒绝覆盖，避免改变现有批次成员。

在仓库根目录执行（PowerShell）：

```powershell
python roboego_hand_vis/download_data/jianzhi/inventory.py `
  --ks3-tool C:\Users\22563\Desktop\utils\ks3_internal\ks3util.exe `
  --ks3-config C:\Users\22563\Desktop\utils\ks3_internal\ks3util-config `
  --output-dir C:\Users\22563\Desktop\data\jianzhi\batches_2026-11-05
```

2026-11-05 实际清单：30,605 条，共 307 批，前 306 批各 100 条，末批 5 条。business 13,980 条，domestic_services 11,117 条，industry 5,508 条。不按日期筛选。

下载使用 `download_data/base` 的通用单列 CSV 下载器，不限制数据集名称。`--jobs` 控制 KS3 内部并发，默认 64。例如在仓库根目录下载第一批：

```powershell
python -m roboego_hand_vis.download_data.base.run `
  --batch-csv C:\Users\22563\Desktop\data\jianzhi\batches_2026-11-05\batch_0000001.csv `
  --ks3-tool C:\Users\22563\Desktop\utils\ks3_internal\ks3util.exe `
  --ks3-config C:\Users\22563\Desktop\utils\ks3_internal\ks3util-config `
  --output-dir C:\Users\22563\Desktop\data\jianzhi\batch_0000001
```

每条数据会下载到 `output-dir/<32位ID>/`。生成清单不下载数据文件。
