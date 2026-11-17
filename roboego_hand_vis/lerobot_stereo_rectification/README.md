# LeRobot 去畸变与立体矫正

按数据来源分级实现：

- [L1](L1/README.md)：L1 头环数据去畸变与立体矫正。
- [L3](L3/README.md)：L3 头环左右目去畸变与立体矫正，按 episode 读取 JSONL 标定。
- [jianzhi](jianzhi/README.md)：简智前向双目，转换 DS 参数并由 body 外参推导左右目变换。
- [archebase](archebase/README.md)：Archebase 鱼眼图像去畸变，同时生成对应左图的双手 3D/2D 点。

```bash
python -m roboego_hand_vis.lerobot_stereo_rectification.L1.run \
  --source /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001 \
  --workers 4
```
