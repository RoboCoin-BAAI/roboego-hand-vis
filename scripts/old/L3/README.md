# L3 流水线

入口 run.sh，配置 pipeline.conf。共有六阶段：download、qc、rectify、depth、vio、review。
没有 convert，不读取 L1 转换凭据。

```bash
bash scripts/old/L3/run.sh batch_0000001
```

stage.py 在 QC 前从 batch CSV 建立 session 清单，之后仅放行 QC 和去畸变通过的数据。
后续深度、VIO、标注使用同一份去畸变通过清单。
默认质检仍要求四路头环视频，缺少 bleft 会被排除；本次目录整理不改变该规则。
标注预算按每个视频计算，失败不会自动重试付费阶段。
查看 scripts/README.md 的总流程和迁移说明。
