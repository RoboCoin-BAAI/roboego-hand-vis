# Archebase CSV 批次

列举 `ks3://baai-eai-pwm/archebase_1000/` 的对象，以同时包含非空 `data.mcap` 和 `annotation.json` 的目录作为一条数据。发现缺失配对文件时停止，不输出不完整清单。

复用简智的批次写入逻辑：按完整路径排序，每批 100 条，单列表头 `目标路径`，UTF-8 BOM 编码，文件名 `batch_0000001.csv`。已有清单拒绝覆盖。

在仓库根目录运行：

```powershell
python roboego_hand_vis/download_data/archebase/inventory.py `
  --ks3-tool C:/Users/22563/Desktop/utils/ks3_internal/ks3util.exe `
  --ks3-config C:/Users/22563/Desktop/utils/ks3_internal/ks3util-config `
  --output-dir C:/Users/22563/Desktop/data/archebase/batches_2026-11-06
```

2026-11-06 清单：21 个场景，18,342 条数据，184 个 CSV；前 183 批各 100 条，末批 42 条。`inventory.json` 保存完整路径、场景计数和批次数量。此操作只列举对象，不下载数据内容。
