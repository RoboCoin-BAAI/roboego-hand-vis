# 数据下载

- L1/run.py：L1 batch CSV 下载。
- L3/run.py：L3 batch CSV 下载，默认全批次自动分 batch 目录。
- L1/cli.py：原 L1 数据库/直接 KS3 路径下载入口，使用 python -m roboego_hand_vis.download_data.L1.cli。
- L1/database_adapter.py：数据库任务领取与回写。
- downloader.py：共用 KS3 下载引擎（Linux 文件锁）。

CSV 下载入口的服务器命令保留在各 run.py 末尾。
详细数据库下载说明见 L1/README.md。
