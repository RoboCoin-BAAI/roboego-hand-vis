# V2 千问时序标注实验

新版独立放在此目录，旧 `qwen_annotation` 代码、原命令、数据库和共享盘流程不修改。
仍使用千问接口、任务信息翻译和原来的帧字段，新增时间上下文与可控并发，并可选择独立图片或拼图。
此入口处理一个视频，不更新数据库状态；默认输出路径由命令指定，建议放到新的实验目录。

## 同一视频对比两种输入形式

在项目根目录运行，设置 `recommended.yaml` 的 `api_key_env` 对应环境变量，例如：

```bash
export DASHSCOPE_API_KEY="你的key"
```

独立图片：

```bash
python -m roboego_hand_vis.qwen_annotation_v2.run \
  --video ./data/l1/session_xxx/videos/left.mkv \
  --output_json ./data/l1/session_xxx/annotation/qwen_v2_individual/qwen.json \
  --image-mode individual
```

带时间标签拼图：

```bash
python -m roboego_hand_vis.qwen_annotation_v2.run \
  --video ./data/l1/session_xxx/videos/left.mkv \
  --output_json ./data/l1/session_xxx/annotation/qwen_v2_grid/qwen.json \
  --image-mode grid
```

两种模式必须使用不同输出目录；抽帧、raw、checkpoint、batch、chunk 默认都位于对应 JSON 的父目录。
自定义配置用 `--config`，自定义目录参数也必须分开，避免混用旧结果。
只重跑相同目录会复用已完成批次，不能用其缓存耗时衡量真实模型提速。
对照旧版时，也将旧版 `--output_json` 指向新的第三个实验目录，保留原有业务产出。

## 推荐参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `fps` | 1 | 每秒采样数量，先保持不变便于对比 |
| `batch_size` | 10 | 每批需要输出标签的目标帧数 |
| `overlap_frames` | 2 | 前后各补最多两帧，只提供时序上下文 |
| `max_concurrency` | 3 | 同时在途的标注请求数；各进程分别计算 |
| `image_mode` | `individual` | 独立图片，或 `grid` 拼图 |
| `grid_columns` | 4 | 每张横向拼图最多包含四个连续时刻 |
| `grid_cell_width` | 512 | 拼图每格宽度，保留比例，不放大原图 |

`--max-concurrency`、`--overlap-frames`、`--batch-size` 和 `--image-mode` 可覆盖 YAML。
例如 `--max-concurrency 1` 可测试串行耗时；改变并发数允许续跑，改变图片模式、采样、重叠或核心批次划分需新的输出目录。
调大 batch_size 会增加单次输出量，可能增加延迟或截断风险，先保持 10。

例如目标时间为 10–19 秒的批次，还可以看到 8、9、20、21 秒作为上下文。
提示词强调同一个动作可能跨越请求边界，不因窗口起点/终点推断动作重新开始或结束。
只保存 10–19 秒的结果，其他时刻标签即使返回也不合并进本批。有效窗口间的断开不提供跨窗口上下文。

拼图从左到右按时间排列，顶部和请求文本都带时间标签。模型仍返回逐时间点的原帧字段，
不为整张拼图生成一条标签。拼图的手部/物体细节可能比独立图片少；是否改善效果及视觉 token 用量需实际对比。
两种模式都是抽样画面，不包含两帧之间的完整运动，不能保证恢复亚秒动作顺序。

## 输出与恢复

请求可以乱序结束；batch、checkpoint、chunk 由主线程统一写，最终标签按原视频相对时间排序。
返回缺帧、重复时间点或窗口外时间时，会进入现有校验重试流程；重试耗尽的批次保留错误并标为失败。
其他成功批次继续保存，下次同设置运行只补缺失批次。

`raw/run_settings.json`（默认位于 `qwen_frame_annotation_raw/`）绑定视频身份、提示词及采样设置，
不保存 API key。旧版没有此设置记录的产出不自动导入 V2；旧结果仍可以由旧流程读取。
原始响应恢复也按批次目标时间过滤，不会重复加入上下文帧或把缺帧响应标成已完成。
任务 TXT/YAML 的翻译和物体闭集约束沿用旧逻辑，元数据变化需新目录。

整数秒采样保留旧 `frame_t000001.jpg` 文件名，小数秒使用如 `frame_t000001_500000.jpg` 的唯一文件名。
未修改旧字段整理代码；它仍按旧整数秒文件名查找图片，所以 V2 先保持 1 FPS 做比较，
小数秒标注及新目录的后续整理接入应单独配置/适配。

命令结束的 summary 提供 `timings.extract_seconds`、`annotation_seconds`、`elapsed_seconds` 和 API `usage`。
`request_seconds_sum` 是各请求（含重试）的耗时之和，并发时可大于总运行耗时。
对比时使用同一视频、采样率、模型和核心批次大小，关注总耗时、失败数、token 用量和跨批次动作一致性。
这里没有自动增加“平滑”动作标签或修改模型输出语义。

设计参考 Argus 的[并发任务](https://github.com/Pantheon-Industries-Inc/argus/blob/main/label/harness.py)、
[分段时间线](https://github.com/Pantheon-Industries-Inc/argus/blob/main/label/pieces.py)和
[按时刻排列的网格图](https://github.com/Pantheon-Industries-Inc/argus/blob/main/label/frames.py)。
实现为本项目新增代码，没有引入 Argus 的模型/API 或数据格式依赖。
