> Legacy batch workflows are archived in [old/](old/README.md). Use [pipeline/](pipeline/README.md) for the current CPU/GPU workflow.

# L1 / L3 流水线

推荐使用新的 [CPU/GPU 独立入口](pipeline/README.md)：各等级目录的 pipeline_cpu.sh、pipeline_gpu.sh 和 pipeline.yaml。
下文 run.sh/pipeline.conf 保留为旧单机整批入口，不与新调度器混用。

入口统一分目录，不再使用 scripts/l1_pipeline.sh 等平铺名称。

| 环节 | L1 | L3 |
| --- | --- | --- |
| 下载 | download_data/L1/run.py | download_data/L3/run.py |
| 转换 | Multi-Frequency-LeRobot/format_convert/L1/batch_convert.py | 无，已是 LeRobot |
| 质检 | lerobot_headband_integrity/L1/run.py | lerobot_headband_integrity/L3/run.py |
| 去畸变 | lerobot_stereo_rectification/L1/run.py | lerobot_stereo_rectification/L3/run.py |
| 深度 | lerobot_fast_foundation_depth/L1/run.py | lerobot_fast_foundation_depth/L3/run.py |
| VIO | lerobot_vio/L1/run.py | lerobot_vio/L3/run.py |
| 标注 | argus/review/L1/run.py | argus/review/L3/run.py |

前六类模块路径相对 roboego_hand_vis；转换和标注位于相邻仓库。
通用数学、编码和后端实现可复用，各等级入口、数据加载和配置分开。

```bash
bash scripts/old/L1/run.sh batch_0000001
bash scripts/old/L3/run.sh batch_0000001
```

运行前检查各目录 pipeline.conf，并在环境设置模型接口密钥；不带 batch 参数将处理全部 CSV。
L1_CONFIG / L3_CONFIG 可指定替代配置文件。脚本是真实执行入口，会下载、计算、调用标注接口；
本次整理没有执行服务器流水线。

L1：下载 → 转换 → 质检 → 去畸变 → 深度/VIO/标注并行。
L3：下载 → 质检 → 去畸变 → 深度/VIO/标注并行。
阶段清单隔离失败 session，保存命令、日志、状态与筛选报告。
L3 默认 CSV 与下载根目录均为 data/L3_lerobot，下载至 batch_*/session。

目录调整不移动数据。原服务端路径需同步更新；同时同步 argus 的 review/L1 和 review/L3。
流水线状态哈希包含脚本与配置，改名后可能产生新状态目录并重新执行阶段，
不要把新入口指向仍有旧流水线运行的数据根目录。

各目录 report.py 用于只读汇总；L1/review_retry.sh 用于已有 L1 标注失败任务重试。
共用 OpenVINS 构建步骤目前复用 L1/vio_setup.sh，L3 不增加转换阶段。
