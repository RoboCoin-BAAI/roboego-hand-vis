# LeRobot 质检

- L1/run.py：L1 头环质检及同 cp 标定补齐，见 L1/README.md。
- L3/run.py：L3 四路头环、头环 IMU、音频和 JSONL 标定，见 L3/README.md。
- common.py：两个等级共用的数据布局和报告工具，不是独立入口。

目录名统一大写 L1、L3。原根目录 run.py 已迁移至 L1/run.py，原 l3 已迁移至 L3。
