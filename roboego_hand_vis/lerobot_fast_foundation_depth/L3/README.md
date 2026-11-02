# L3 Fast-FoundationStereo 深度

先执行 lerobot_stereo_rectification/L3/run.py。读取：

- observation.images.headband_left_rgb_rectified / headband_right_rgb_rectified 视频。
- 当前 episode 的 calibrations/[chunk_episode_]dataset_manifest.json 中的矫正内参与基线。
- 左目矫正 Parquet，原样保存为 depth/chunk-*/episode_*/source_timestamps.parquet。

不直接用原始 JSONL 鱼眼内参估计矫正图像深度，不把原始与矫正坐标系混用。
视频路径按当前 session 重建，支持跨机器移动；原始文件不修改。
输出沿用 L1 depth/chunk-*/episode_* 格式及 lerobot_depth.json，来源特征记录为 L3 名称。
本阶段不自动注册新的 LeRobot 深度 feature。

```bash
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_fast_foundation_depth/L3/run.py \
  --source /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001 \
  --model-root /share/project/liuyou/humancoin/code/Fast-FoundationStereo \
  --ckpt /share/project/liuyou/humancoin/code/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
  --engine throughput --gpus 0,1,2,3,4,5,6,7 \
  --num-frames auto --depth-format hevc12 \
  --ffmpeg /share/project/liuyou/humancoin/envs/ego/bin/ffmpeg
```

--dry-run 仅验证路径和 manifest，不加载 CUDA 模型。
推荐配置为本目录 recommended.yaml，复用 L1 的模型参数、并行调度、编码和推理实现。
支持 throughput/legacy、GPU 列表、files/hevc12、预览、区间及右帧偏移选项。
以同帧号读取双目图像；未自动验证或修正时间同步。
完整推理需要服务器上的 CUDA、模型源码和权重；CPU 测试只验证加载及与矫正输出的衔接。
