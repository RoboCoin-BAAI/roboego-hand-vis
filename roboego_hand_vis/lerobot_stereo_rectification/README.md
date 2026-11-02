# LeRobot 去畸变与立体矫正

按数据来源分级实现：

- [L1](L1/README.md)：L1 头环数据去畸变与立体矫正。
- [L3](L3/README.md)：L3 头环左右目去畸变与立体矫正，按 episode 读取 JSONL 标定。

```bash
python -m roboego_hand_vis.lerobot_stereo_rectification.L1.run \
  --source /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001 \
  --workers 4
```
