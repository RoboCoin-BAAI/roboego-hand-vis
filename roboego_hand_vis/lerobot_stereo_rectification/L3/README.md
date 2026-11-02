# L3 头环左右目去畸变与立体矫正

读取原始 observation.images.headband_left_rgb / headband_right_rgb。
支持单 session、batch 或递归数据根目录；不依赖第四路头环视频。

```bash
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/roboego_hand_vis/lerobot_stereo_rectification/L3/run.py \
  --source /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001 \
  --ffmpeg /share/project/liuyou/humancoin/envs/ego/bin/ffmpeg \
  --workers 4
```

在项目根目录也可使用 `python -m roboego_hand_vis.lerobot_stereo_rectification.L3.run`。
依赖 numpy、opencv-python、pyarrow、PyYAML，以及支持 libx264 的 ffmpeg。
--dry-run 读取 JSONL 标定和视频尺寸，不编码或写文件。
--balance 范围 0 到 1，默认 0；--sessions-file 接受所选 session 路径 JSON 列表。
默认验证已有结果后复用；--no-skip-existing 强制重算。
每路编码首帧及每 300 帧打印进度，每个 session 完成后打印结果。

## L3 标定适配

L1 从 calibrations/camera-camchain.yaml 读取相机；L3 从
meta/calibrations.jsonl 按 episode_index 唯一匹配，读取
calibrations.ego_headband.camera-camchain。不使用 camera_imu 分支替代。
左右相机通过 rostopic 的 left/right 段映射；样本为 cam0/cam1。
要求右相机是左相机之后的 Kalibr 相机，右 T_cn_cnm1 为左到右变换。
只接受 pinhole + equidistant、合法旋转和非零基线，左右标定尺寸必须一致。
不会把其他 episode 的标定静默用于当前 episode，也不会修改或生成原始 JSONL 标定。

复用 L1 的几何模型、视频编码、发布回滚和结果复用校验。
默认居中裁剪输入到标定尺寸，例如 1920×1200 → 1920×1088，上下各 56 像素。
这是与 L1 相同的居中裁剪假设，未从 L3 元数据证明真实裁剪位置。
使用 --no-center-crop-calibration 可要求视频与标定尺寸完全一致。
不做缩放、不调整输入标定主点。矫正后裁剪左右有效区域交集，更新 P1/P2/Q 和输出内参。

## 输出与帧策略

- videos/chunk-*/observation.images.headband_left_rgb_rectified/episode_*.mp4
- videos/chunk-*/observation.images.headband_right_rgb_rectified/episode_*.mp4
- data/chunk-*/episode_*/observation.images.headband_left_rgb_rectified.parquet
- data/chunk-*/episode_*/observation.images.headband_right_rgb_rectified.parquet
- calibrations/rectify_params.yaml、frame_mapping.parquet、dataset_manifest.json
- meta/info.json 注册新增相机特征，更新视频文件数，保留原特征。

多 episode 的参数、映射和 manifest 带 chunk/episode 文件名前缀。
manifest 保存 JSONL 来源、episode、标定分支和内容 SHA256，供结果复用校验。
左右视频分别按源帧序全部处理，可有不同 FPS 和帧数；不检查时间戳、不重新配对或补删帧。
时间戳 Parquet 原样复制。manifest 标记 synchronization_checked: false。
原始视频与标定不覆盖；输出先在隐藏临时目录编码，成功后发布，失败保留旧结果。

当前已用真实 L3 样本验证标定加载和裁剪预览，并用小视频验证完整编码、输出元数据和复用流程。
尚未对完整真实 L3 视频执行编码验证；几何矫正也不代表左右视频已时间同步。
