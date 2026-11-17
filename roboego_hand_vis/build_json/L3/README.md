# L3 精简 readme.json

在仓库根目录运行：

```powershell
python roboego_hand_vis/build_json/L3/build_readme.py --root "C:\Users\22563\Desktop\data\L3-lerobot\L3-001-2026-10-26-13-21-17"
```

`--root` 支持单条 session、批次或数据集目录，输出至各 session 的 `meta/readme.json`。可用 `--context` 指定主相机等技术配置。任务名、描述、场景、动作和物品只读取 annotation/episode_*_structured.json，不使用采集元数据或 context 中的语义字段回填。缺失语义保留 null、unknown 或空数组；英文命名沿用结构化标注。

L3 CPU pipeline 已接入 readme 阶段，顺序为 download → qc → rectify → review → points → readme。readme 依赖 rectify、review、points 成功，不等待 GPU depth；深度未生成时不填充深度流。深度报告变化后，下次运行 CPU pipeline 会更新 readme。

顶层字段与 base 精简 readme 对齐，包含 common_task，保留 L3 的手套、手腕轨迹、脑电、肌电等模态。actions 只使用结构化标注的 actions.vocabulary，不从 subtask 回填。帧数优先累计 master_index.parquet 实际行数，缺少索引时回退 info.total_frames。

Parquet 模态仅包含找到文件的流，空文件保留零样本数。深度额外读取 depth/chunk-*/*/lerobot_depth.json，兼容旧版单条报告和 schema_version=2 的多 episode 报告；仅纳入存储文件实际存在的深度流，输出尺寸、帧数、采样率、单位、无效值及编码格式。深度尚未生成时可以先构建，生成后需重新构建 readme。

场景类型共用 base 的 HumanCOIN V1 推断规则：例如 commercial_service/hair_salon 推断为 indoor，public_outdoor/farm 推断为 outdoor，无法判断的区域保留 unknown。这是典型功能场景的推断，不代表实地位置已核实。category 和 area（输出为 scene.name）保留结构化标注的值。

EEG 排除已命名辅助通道。相机数量表示 info 声明的视频流数量，名称中的 views 不代表立体相机目数。手部采集类型有手套流时为 glove，否则 unknown。

dataset_name 不拼接原文件夹名，dataset_name 和 task_name 的分隔符统一为下划线。文件统计排除 meta/readme.json 及其临时文件，重复生成保持稳定。原始标注不变。
