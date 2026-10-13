# L1 OpenVINS 数据库入口

范围：保留原手动 CLI、准备脚本及 OpenVINS 算法，独立数据库入口固定 rectified，按既有 YAML 依赖等待立体矫正。

- [x] DataService 原子领取 pending、短事务 processing 和 completed/failed 回写，不作为阈值门禁。
- [x] 二进制、系统 Python 和时间参数预检；每 session 路径与输入文件边界检查。
- [x] 复用原 build_openvins_command，独立进程组日志与中断清理。
- [x] 清单、报告及轨迹必须本次更新；来源匹配、initialized、正 pose_count、JSONL 数量及结构/数值校验。
- [x] 不要求初始化前每个视频帧都有位姿，不裁减视频、不放宽原 IMU 时间覆盖检查。
- [x] 使用说明及限定只读代码审查。

验证：全量 327 项测试通过（pytest --import-mode=importlib，临时 PostgreSQL），包含阶段依赖/并发、空轨迹、缺记录、非有限数值、旧输出不复用、中断回写和日志子进程替身。手动与数据库 CLI help 可用，git diff --check 通过。未运行真实 OpenVINS 业务数据或修改业务数据库；vio_status 已存在，无需迁移。
