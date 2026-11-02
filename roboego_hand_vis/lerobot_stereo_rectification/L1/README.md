# L1 LeRobot 去畸变与立体矫正

独立入口，原有 `stereo_rectification` 不变，复用其 equidistant 数学模型。
左右视频分别按原始帧序全部矫正，不检查时间戳、不重新配对、
不裁掉首尾、不补帧、不删帧，也不要求左右帧数或 FPS 相同。
同步检查及修正由独立阶段负责。

需要 `numpy opencv-python pyarrow PyYAML` 和支持 libx264 的 `ffmpeg`，不需要 GPU。
在项目根目录执行：

```bash
python -m roboego_hand_vis.lerobot_stereo_rectification.L1.run \
  --source /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001 \
  --workers 4
```

`--source` 支持单数据集或批次，`--workers` 控制 CPU 进程数。
`--dry-run` 读取标定及视频尺寸，不编码、不写文件。
`--ffmpeg` 指定编码器路径；`--balance` 为 0 到 1，默认 0。

## 标定与尺寸

支持标准相机名，也兼容原始 `cam0/cam1` 键，通过 `rostopic` 确定左右相机。
原始 Kalibr 格式要求右相机紧接左相机，右目 `T_cn_cnm1` 表示左到右变换；
标准命名要求 `extrinsics_reference_camera: head_left`。
只支持 pinhole + equidistant，外参单位为米。
保留模型、矩阵、尺寸及视频读写等必要检查。

与原有模块一致，默认自动居中裁剪到标定尺寸，例如 1920×1200 上下各裁掉
56 像素得到 1920×1088。尺寸一致时不裁剪，视频小于标定时报错。
差值为奇数时右侧/下侧多裁一个像素。不缩放、不调整标定主点，
假设标定图像对应原图的居中裁剪区域，输入 ROI 会保存到参数文件。
可用 `--no-center-crop-calibration` 关闭自动裁剪，关闭后尺寸不一致会报错。
随后复用原实现的有效区域算法，计算左右 ROI 交集并裁剪输出黑边，宽高向下取偶数。
该算法使用行列有效像素比例阈值 0.95，并非保证每个像素都无黑边。
裁剪同步更新 P1/P2 的主点、Q、manifest 内参及最终输出尺寸。

## 输出

```text
<session>/
├── videos/chunk-000/
│   ├── observation.images.head_left/episode_000000.mp4
│   ├── observation.images.head_right/episode_000000.mp4
│   ├── observation.images.head_left_rectified/episode_000000.mp4
│   └── observation.images.head_right_rectified/episode_000000.mp4
├── data/chunk-000/episode_000000/
│   ├── observation.images.head_left_rectified.parquet
│   └── observation.images.head_right_rectified.parquet
├── calibrations/
│   ├── frame_mapping.parquet
│   ├── rectify_params.yaml
│   └── dataset_manifest.json
└── meta/info.json
```

原视频和原相机特征保留，新增两个 `_rectified` 特征。
对应时间戳 Parquet 必须存在，会原样复制为新特征，不进行同步检查或修正。
`meta/info.json` 注册新特征的尺寸、视频信息，并更新实际视频文件总数。
`calibrations/` 下参数包含标准化的校正相机内参、P1/P2/Q、裁剪信息及最终尺寸。
单 episode 直接使用上述文件名；多 episode 使用 `chunk-000_episode_000000_` 等前缀区分，
不增加子目录。
映射中每路源帧号与输出帧号一致；左右长度不同时，短流映射为 null，不补帧。
manifest 分别记录实际帧数及各自 FPS，标记 `synchronization_checked: false`，
保存矫正后的内参和基线，以及输出 ROI 和最终尺寸。
已有校正结果默认覆盖：先在隐藏临时目录完成编码，成功后发布新视频、
时间戳和参数，最后更新 info.json；发布失败时回滚被替换的文件。
编码失败保留旧结果和失败临时目录，不影响原始视频。

后续可把 manifest 交给 `fast_foundation_depth.run`，本阶段不自动运行深度。
现有深度模块按同序号帧读取左右图像，同步检查及修正需由独立阶段完成。
深度属于矫正左目坐标系，不能直接当作原始鱼眼坐标系深度。
