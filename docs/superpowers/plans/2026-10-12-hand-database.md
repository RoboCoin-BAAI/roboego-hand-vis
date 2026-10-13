# L1 原生手部估计数据库入口

范围：保留 hand.run 手动入口及 MINT/EgoHandKit 算法参数，新增独立数据库领取和结果回写，必须等待深度完成。

- [x] DataService claim_hand/finish_hand，pending 原子领取及短事务回写，非门禁 completed/failed。
- [x] 复用 native.execute，输入每 session，输出 session/hand，checkpoint 自动恢复。
- [x] 恢复前校验 checkpoint 的 plan、fingerprint；防止输入更新后旧 delivery 被记为新结果。
- [x] EgoHandKit Python 最终 Parquet/渲染帧数验证、交付哈希检查和输入处理前后一致性检查。
- [x] annotation 中原子完成 receipt，内部保留 native full checkpoint 至 receipt 落盘后再 compact，支持写入/清理故障恢复。
- [x] Linux 子进程组清理与中断状态回写，保留原手动参数、算法与非 Linux 原分支。
- [x] 数据库解锁下游深度归档、使用说明与限定代码审查。

验证：全量 319 项测试通过（pytest --import-mode=importlib，临时 PostgreSQL），包括真实 Parquet/视频校验、旧交付输入改变、缓存复用、receipt 写入失败、cleanup 中断和进程组清理。未运行真实 GPU 推理或写业务数据库。hand_status 已存在，无需迁移。
