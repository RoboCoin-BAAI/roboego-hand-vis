# LeRobot Fast-FoundationStereo 深度估计

- [L1](L1/README.md)：适配 L1 LeRobot 的矫正视频、标定 manifest 和时间戳。
- [L3](L3/README.md)：适配 L3 头环左右矫正视频和 episode manifest。

原 `fast_foundation_depth` 保持不变，新目录复用其模型推理 API。


L1/L3 depth and VIO do not repeat first-frame stereo quality assessment before runtime. Upstream quality control owns that assessment. Required input, calibration, timestamp and file checks remain; VIO trajectory quality is still evaluated after estimation.

## 公共实现

`base/` 保存 L1/L3 共用的输入解析、模型推理、GPU 调度、预览、深度编码和发布实现。L1/L3 的 `compute.py` 仅适配相机名称和各自的 `recommended.yaml`，`run.py` 保留各自的运行入口。模型参数和输出格式保持不变。

L1 下原公共模块保留兼容导入；新增调用应使用 `lerobot_fast_foundation_depth.base`。底层模型 API 继续复用 `fast_foundation_depth`。

默认 HEVC12 发布时同步更新 session 的 `meta/info.json`：注册单通道深度 video feature，记录尺寸、帧率、编码、毫米单位、无效值和掩码要求，并按已注册 feature 的实际 MP4 文件更新 `total_videos`（不计掩码）。视频、掩码、时间戳和 info 使用同一发布事务，info 最后写入。逐帧 files 模式不注册为 video。已有结果可用 `scripts/update_depth_meta.py --root <session> --execute` 补登记。

深度发布和重编码不使用文件锁，也不生成锁文件；数据写入顺序由外部任务编排负责。
