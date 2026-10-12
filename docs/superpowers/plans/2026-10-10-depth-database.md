# L1 深度估计数据库入口

范围：保留原手动命令、推理算法和默认 600 帧配置，新增独立数据库入口。沿用已配置的依赖，先补齐深度前质量门禁。

- [x] 增加 stereo_depth_gate 和 fast_foundation_depth 原子领取与受保护状态回写。
- [x] 门禁采用现有 usable_for_depth：通过 completed、不通过 block、异常 failed。
- [x] 深度估计调用原 GPU 命令，继承 CUDA_VISIBLE_DEVICES，不持有数据库长事务。
- [x] 每条 session 独立路径与日志；完成前检查配置范围内的全部深度输出。
- [x] 中断清理子进程组后回写 failed；原有手动流程不变。
- [x] 新增阶段文档、结果完整性与中断测试、真实 PostgreSQL 依赖与并发测试。
- [x] 完成限定范围只读代码审查，无阻塞问题。

验证：相关阶段和数据库全套测试通过。验证使用临时 PostgreSQL 和模拟推理输出，未执行真实 GPU 推理或修改业务数据库。
