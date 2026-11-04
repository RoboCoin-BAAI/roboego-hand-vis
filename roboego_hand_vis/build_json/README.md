# 自采数据 JSON 字段定义（v1）

先固定字段语义，再从 L1/L2/L3 提取。`build_metadata.py` 读取 LeRobot session，生成描述整条数据的 manifest，不把逐帧数组塞入 JSON，也不修改输入。依赖 Python 3.10+、`pyarrow`。目前验证的是给定 L3；其他布局需另加适配。

## 基本约定

- 英文字段便于程序读取，本文给出中文解释。未知值用 `null`，未知类别用 `unknown`；空数组表示未提取到记录。
- 模态状态：`present` 表示至少一个文件非空，不保证信号有效；`empty` 表示文件存在但零行；`declared_only` 表示只有声明；模态总状态 `not_found` 表示没有找到非空数据，不能据此断定未采集。
- 通道数：无非空信号流记 0；有信号流但不能确定通道语义时记 null。EEG 排除已命名 VEOG/HEOG/EKG/ECG/EMG/Trigger 辅助列；EMG 仅合计独立肌电流，不重复计算 EEG 辅助 EMG。
- 所有路径相对输入 session，只有 `identity.source_path` 为绝对路径。所有原始参数保持来源的单位，不擅自转换。

## 顶层字段

|字段|定义与来源|
|---|---|
|schema_version|本 manifest 的结构版本，与 LeRobot 版本无关|
|dataset_name|`级别_场景_任务名称_主视角简称_主视角宽x高_FPS_几目_EEG通道数_EMG通道数_原文件夹名`；默认主视角简称为 left；未知段为“待确认”，仅显示名称，不重命名目录|
|identity|level 为 L1/L2/L3；original_folder_name 保留原名；source_path 记录输入位置|
|task.scene|type：indoor/outdoor/mixed/unknown；name 优先使用人工填写值，否则使用 collect_info.jsonl 中唯一的非空 scene，source 记录名称来源。模板中的 null 不覆盖采集场景；多个场景不自动选取|
|task.name|短任务名称，人工优先，否则唯一 tasks.jsonl 的 task|
|task.description|任务目标与操作过程，人工优先，否则结构化标注 task_description|
|task.actions|动作集合，默认对 annotation.subtask[].action 去重；不等于数值 action 特征|
|task.objects|实际标注物品及属性，默认来自结构化标注；采集配置物品另保存在 source_metadata.collection 中，不混合去重|
|task.annotation_sources|语义字段的来源文件；自动标注未经人工确认|
|statistics.duration_s|名义时长，主时间轴帧数/FPS，单位秒；不等于任意传感器覆盖时长|
|statistics.frame_count|master_index.parquet 实际总行数，无索引才回退 info.total_frames|
|statistics.master_fps|info.fps；不覆盖各模态原生采样率|
|statistics.episodes|每个主索引的帧数、最大减最小时间戳跨度；跨度通常比名义时长少一帧|
|statistics.size_bytes|输入目录全部普通文件的逻辑字节数，含 partial/cache；不是磁盘占用，也不是 info 中的分块大小配置|
|statistics.file_count|同一范围的文件数|
|cameras.count|info 中 video 特征个数，含声明但缺失的视频；逐相机 status 区分|
|cameras.primary_camera|默认 observation.images.headband_left_rgb_rectified，可通过 context 覆盖；null 使用默认值，不自动使用 master_feature|
|cameras.primary_camera_label|用于 dataset_name 的主视角简称，默认 left|
|cameras.primary_camera_status|主视角流的文件状态；info 未声明默认矫正流时为 not_found，此时 items 不伪造该流，分辨率和 FPS 待确认|
|cameras.synchronization_master|info.master_feature，仅表示同步参考流|
|cameras.view_count|用户命名规则中的“几目”，人工指定；不能将总共 8 个相机直接解释为八目立体相机|
|cameras.items|每个相机 id、宽、高、FPS、视频路径、主视角标志和精简特征参数；不包含内外参详情|
|channel_counts|命名中使用的 EEG、独立 EMG 总通道数及计数口径|
|modalities|下表的固定模态组；空缺模态仍保留定义与状态|
|coordinate_systems|变换方向约定及世界到相机外参；没有证据的变换不填单位矩阵|
|calibration.source|仅记录标定文件相对路径；文件不存在时为 null，不内嵌标定详情|
|source_metadata|原始 tasks、collect_info、episodes 记录，保留来源语义|
|review|发现的问题和待确认项；生成成功不代表这些问题已经解决|

## 模态定义与 L3 映射

每组有 `definition/status/streams`。每条 stream 保存 `feature_key`、精简 `definition`（dtype、shape、fps、window、tolerance 等）、`sample_rate_hz`、`sample_count`（实际 Parquet 行数）、`data_paths`、`status`、`coordinate_frame`、`units`、`provenance_paths`。各模态 definition 和相机 parameters 均省略 `names`、`landmark_order` 列表；完整字段顺序可查输入 meta/info.json。通道计数仍读取原始 names，不受输出精简影响。不同采样率的数据不按 543 帧强行对齐；时间窗口和容差原样保留。

|模态|L3 特征及参考系处理|
|---|---|
|depth 深度数据|包含 depth 的特征；本样本没有找到，不能因为 D455f 硬件存在就声称有深度数据|
|hand_estimation 手部估计|hand_left_points/right_points：每手 21×3，info 明示 OpenVR TrackingUniverseStanding 世界系、米；来源是 MANUS+VIVE+固定安装变换，非裸手估计。安装外参为假设，保留 extrinsic_status 和 provenance 文件引用|
|head_trajectory 头部轨迹|head_pose：xyz+RPY；源 info 未定义参考系/单位时留空，不与相机光心轨迹等同|
|eye_gaze 眼动轨迹|eye_gaze：二维 x/y；不将其猜成三维视线，也不未经确认假设归一化或像素坐标|
|imu|双腕、眼镜、头带各 IMU：ax/ay/az/gx/gy/gz；各自参考系和单位待确认|
|hand_trajectory 手部轨迹|left_wrist_pose/right_wrist_pose：左右腕跟踪器 xyz+RPY；与骨架/21 点分开|
|emg 肌电|left_wrist_emg/right_wrist_emg，各 8 列、各自约 2000Hz；电压单位待确认|
|eeg 脑电|eeg 为 132 信号列，其中 128 个非辅助列、4 个辅助列；eeg_impedance 的 136 维不能当作脑电通道数|
|glove 手套|hand_pose：40 关节参数；hand_skeleton：50×6 节点。硬件记录是 MANUS；原始骨架坐标系和单位不由节点数猜测|
|bare_hand_estimation 裸手估计|仅根据 context.bare_hand_features 明确指定；L3 默认没有，不把手套衍生 21 点归为裸手|

本仓库 `lerobot_hand_pose/L3/WORLD_POINTS.md` 和采集打包脚本提供了 Tracker 的 OpenVR、米、度及 `Rz(yaw) @ Ry(pitch) @ Rx(roll)` 解释，但生成器优先遵循数据自带声明；尚未将这些 L3 特定解释推广到 L1/L2。接下来应逐设备补齐坐标轴、单位、眼动参考图像、时间原点与同步方法。

## 相机内外参

JSON 仅保留 `calibration.source` 标定文件路径。相机模型、内参数值、畸变系数、外参矩阵和相机 IMU 时间偏移等详情直接查阅原始 `meta/calibrations.jsonl`，不复制到本 JSON，也不展开逐相机标定引用。

## 使用

在仓库根目录运行：

```powershell
python roboego_hand_vis/build_json/build_metadata.py "C:\Users\22563\Desktop\data\L3-lerobot\L3-001-2026-10-26-11-07-59" --output roboego_hand_vis/build_json/L3/L3-001-2026-10-26-11-07-59.json
```

复制 `context.template.json` 后填写已确认语义，增加 `--context 路径` 重新生成。模板中的 null 字段表示待确认。不要直接把模板中的空 actions/objects 当作自动结果；需要覆盖时才新增这两个键。生成文件必须在输入目录之外，避免把上次输出计入数据大小。

本样本名义时长 18.1 秒，543 帧。场景名称从 collect_info.jsonl 读取为“餐桌”，同步用于 dataset_name；“餐桌”不直接推断为室内采集。主视角默认 headband_left_rgb_rectified，命名简称 left。info 尚未声明该矫正流时，不使用原始鱼眼参数代替。结构化标注含 19 秒结束的片段，生成器会提出复核。室内/室外和几目仍需确认。
