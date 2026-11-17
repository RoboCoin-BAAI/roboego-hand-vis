# Archebase batch 流水线

参考简智 pipeline，依次处理每个 batch：**下载原始 MCAP → 多频 LeRobot 转换 → 图像与双手点去畸变 → 文本模型清洗源标注 → 上传 → 校验后删除本地数据**。不运行 Fast Foundation 深度。

服务器在 roboego-hand-vis 仓库根目录运行：

```bash
/share/project/liuyou/humancoin/envs/ego/bin/python scripts/archebase/pipeline.py
```

默认路径：

| 参数 | 默认值 |
| --- | --- |
| `--batch-dir` | `/share/project/liuyou/humancoin/data/archebase_batches` |
| `--local-root` | `/home/archebase` |
| `--state-root` | `/share/project/liuyou/humancoin/data/archebase_pipeline_state` |
| `--converter-repo` | `/share/project/liuyou/humancoin/code/Multi-Frequency-LeRobot` |
| `--argus-repo` | `/share/project/liuyou/humancoin/code/argus` |
| `--destination` | `ks3://baai-eai-pwm/archebase_lerobot/`（固定） |
| `--model` | `qwen3.6-plus` |

KS3 和 FFmpeg 默认使用与简智相同的服务器工具路径。需要同步三个仓库的新代码，并在运行 Python 环境安装 Archebase 转换、OpenCV 和 Argus 文本清洗依赖。模型沿用 Argus 的 XMSXB/DashScope 配置。默认模型需要 `XMSXB_API_KEY`，可选 `XMSXB_BASE_URL`；密钥不要写入脚本。启动时检查配置，缺失时在下载前报错。

先查看一条计划：

```bash
python scripts/archebase/pipeline.py --session-id 186460 --dry-run
```

只处理一个 batch：

```bash
python scripts/archebase/pipeline.py \
  --batch-csv /share/project/liuyou/humancoin/data/archebase_batches/batch_0000001.csv
```

默认阶段屏障和并发为：整批下载 4 进程 → 整批处理 4 进程（每条依次转换、去畸变、模型清洗）→ 整批上传 4 进程 → 远端复查与本地清理。可通过 `--download-workers`、`--prepare-workers`、`--upload-workers` 调整。失败项写入 skipped.json，保留本地数据，后续阶段和后续运行不自动重试。

本地结构为 `local-root/<batch>/<数字ID>/raw/` 与 `dataset/`。只下载每条的 `data.mcap`、`annotation.json`，使用 KS3 传输校验。转换只转封装，不额外解码质检；去畸变阶段必须解码、remap、重新编码。当前是各目独立去畸变，不宣称双目极线对齐。

标注使用转换阶段的 `episode_000000_source.json` 通过 `argus.review.archebase` 对源文本清洗，不重新用视频标注。模型时间不可更改，不用固定翻译词表。调用结果通过 Argus 回执校验后发布到 `dataset/annotation/episode_000000_structured.json`，同步英文任务名到 tasks/episodes。请求、响应、claim 和日志持久化在 state-root 下；超时不自动重发，便于核对计费与离线恢复。

上传整个 `dataset/` 到 `ks3://baai-eai-pwm/archebase_lerobot/<batch>/<数字ID>/`，包含原始及去畸变 MP4、Parquet、标定、meta、原文、中间格式与最终英文标注。**原始 data.mcap 不上传。** 不根据模型生成的任务名改变路径。

删除前必须已有上传成功收据，再次检查所有远端文件名和大小，并核对本地 raw/dataset 完整文件清单未变化。之后仅删除本流程拥有的 `local-root/<batch>/<数字ID>/`，状态和模型调用记录保留。远端复查是大小校验，不是逐文件下载验哈希。中断可从持久化阶段继续；已有模型结果按调用回执复用。路径、模型或处理配置变化时须使用新 state-root，不接管未登记的本地目录。

`--dry-run` 不下载、不调用模型、不上传、不删除。本脚本不会自行在服务器部署或启动任务。
