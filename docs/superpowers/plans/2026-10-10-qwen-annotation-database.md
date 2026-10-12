# L1 千问标注数据库入口

新增独立 qwen_annotation.database_run，复用现有左目标注算法与 checkpoint。原单视频入口和配置不修改；非门禁，正常完整完成为 completed，部分批次失败或异常为 failed。无新增字段/迁移。

- [x] 下载、基础质检、视频质检 completed 才能原子领取标注 pending；不依赖矫正或手部检测。
- [x] API key 和 prompt 预检，不打印凭据；短事务，标注期间无数据库连接占用。
- [x] 逐 session 生成 frames/raw/checkpoint/batch/chunk/output 路径，保留 resume。拒绝固定共享存储路径及 windows_json。
- [x] 检查 summary 和报告 complete、失败批次为零、非空帧与来源匹配后才写 completed。
- [x] 中断/异常回写 failed，回写失败停止领取。
- [x] 并发、前置门禁、状态隔离、路径、数量、部分失败、预检、产物及连接释放测试和简洁文档。

验证使用隔离 PostgreSQL schema 与模拟 API/报告，不修改业务数据库，不访问真实千问服务。

验证结果：相关测试共 130 项通过，包含报告与汇总不一致/来源错误拒绝完成。独立只读代码审查未发现阻塞缺陷；原标注入口、算法和配置无差异。
