# Archebase 图像与手部点去畸变

输入为转换后的单个 Archebase LeRobot 数据集。读取 `calibrations/camera_info.json` 的 equidistant K/D，为左右目分别生成虚拟针孔相机。当前为**各目独立去畸变**，不宣称双目极线对齐，也不生成用于深度计算的 Q 或基线。原始 CameraInfo 的 P 不作为原始相机间外参。

```powershell
python -m roboego_hand_vis.lerobot_stereo_rectification.archebase.run `
  --source D:/data/archebase/186460_lerobot
```

使用 `--dry-run` 查看参数，`--balance 0..1` 调整视野。FFmpeg 可通过 `--ffmpeg` 指定，否则使用 PATH 或 imageio-ffmpeg。依赖 numpy、OpenCV、PyArrow；原始特征和视频保留，新增特征如下：

- `observation.images.head_left_undistorted` / `head_right_undistorted`：相机原始光轴保持不变，R=I，使用新内参 K_new 和零畸变模型。
- `observation.hand_left_points_undistorted` / `hand_right_points_undistorted`：**两只手均在左相机坐标系中**，变换为 `X_new = R_left @ X_source`。当前 R_left=I，因此 3D 数值无需改变。3D 点本身没有镜头畸变。
- `observation.hand_left_points_2d_undistorted` / `hand_right_points_2d_undistorted`：由上述 3D 点通过左目 K_new 投影为 UV，与去畸变后的左图对应。不能把右手误当作右相机坐标。相机后方或非有限点输出 NaN，图像范围外坐标保留。不额外存储 validity 特征。

图像与手部共享同一 R/K_new，不通过重新拟合 2D 坐标处理。各流时间戳、点顺序和行数保持原样，不做插值。原始手部四元数在 R=I 时仍适用于相同光轴坐标。未来若加入真正的双目旋转校正，3D 点、关节方向和 2D 投影必须一起使用 R_left 更新。

图像去畸变需要解码、重映射和重新编码；复用共享 Encoder：H264、veryfast、CRF18、4 线程、无 B 帧。不进行额外的完整解码质检。处理期间的帧数必须与源相机时间戳数量一致。

输出到同一数据集，新增元信息和统计随文件一起发布；失败时保留 `.archebase-undistortion.partial-*`。已有去畸变特征时拒绝覆盖。`calibrations/archebase_undistortion.json` 保存各目 R/K_new、分辨率、标定哈希和帧数；`preview/episode_000000_hands_undistorted.jpg` 提供双手首帧叠加预览。

目前不接受时间变化的标定。不提供到右图的手部投影，后者需要经确认的原始左右相机外参。
