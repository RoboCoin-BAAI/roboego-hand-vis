# 通用 LeRobot readme 生成器

复用原光轮实现，输出字段和统计规则不变。无需模型调用，生成正式文件 `meta/readme.json`。读取视频容器帧数，不逐帧解码。

```powershell
python -m roboego_hand_vis.build_json.base.build_readme --root D:\data\jianzhi\001301e26b8c4ace844c7b2d12eacf9d
```

也支持直接运行 `python roboego_hand_vis/build_json/base/build_readme.py --root ...`。root 支持单数据集、batch 或包含 batch_* 的目录。`--level L1|L2|L3` 默认 L1。

通用信息来自 meta/info.json、meta/tasks.jsonl、master_index.parquet、实际视频与 Parquet、annotation/episode_*_structured.json。场景类型使用 HumanCOIN V1 对应规则；光轮旧入口继续转调本实现。

物品汇总各标注文件中的 objects[].name，按名称去重，仅输出 name，不输出属性。
任务大类 common_task_name 读取排序后第一个结构化标注文件的同名字段；缺失时为 null。

本次仅抽取共用实现：相机 count 仍为视频特征数量，包含校正派生流；MANO 尚未纳入模态摘要；不自动推断缺失坐标约定。原 L3 专用生成器及简智处理流水线未切换。写入使用同目录原子替换，成功后不保留临时文件。
