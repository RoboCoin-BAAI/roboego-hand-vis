# 千问逐帧标注

对原始左目视频抽帧，通过配置的千问服务生成语义标注。此阶段不作为门禁。

## 标注指定视频

```bash
export DASHSCOPE_API_KEY="你的key"

python -m roboego_hand_vis.qwen_annotation.run \
  --video /path/to/session/videos/left.mkv \
  --output_json /path/to/session/annotation/qwen_frame_annotations_left.json
```

模型、服务地址、采样和重试参数见 `recommended.yaml`，可通过 `--config` 指定其他配置。此命令不回写数据库。

## 自动使用任务信息

输入为 `<session>/videos/left.mkv` 时，自动读取 `<session>/任务详情.txt` 和 `<session>/*_config.yaml`；其他视频路径在视频所在目录查找。单视频和数据库入口使用相同逻辑，不需要增加参数。

- TXT 提供任务名称、完整的多行任务内容和任务道具；YAML 提供 `scene.name`、`scene.notes`、`scene.objects`。优先使用 YAML 物体清单，按 `lib_name` 合并实例；缺少 `lib_name` 时使用 `name`。例如 `砂纸`、`砂纸_c`、`砂纸_c2` 的 `lib_name` 都是“砂纸”，只保留一个类别。颜色和材质用于辅助翻译，渲染和随机布局参数不传给模型。
- 在抽帧标注之前，使用同一千问服务进行一次文本翻译，生成英文场景、详细的总体任务表述及英文物体名称。模型必须保留各步骤和任务细节，不能只输出任务标题。例如场景为 `kitchen stovetop`，任务表述为 `Align and stack sandpaper sheets of different grit sizes, then place them neatly into the storage compartment.`，物体为 `sandpaper` 和 `yellow plastic storage box`。实际措辞由模型翻译后固定缓存。
- 每个批次都携带相同的英文上下文。`interaction_objects`、`attention_objects`、`task_objects` 和 `interaction_object_attributes[].name` 只能使用缓存清单中的精确名称。活动描述需复用交互物体名称。结构化字段出现清单外名称时校验失败并重试；重试耗尽的批次标为失败。自由文本中的其他任务物体名称由提示词约束，程序不做自然语言语义判定。
- 总体任务是预期目标，动作和可见物体仍以画面为准。无法匹配清单的物体不能强行归类或新增名称；模型应在 notes 中说明无法识别并提高不确定性。如果只有清单外物体交互，`hand_object_interaction=false` 表示没有确认的清单内物体交互，手部可见性仍按实际画面标注。背景和隐私内容仍允许描述。

翻译缓存保存在 `annotation/qwen_frame_annotation_raw/task_context.json`（自定义 raw 目录时随该目录）。最终报告、checkpoint、batch 和 chunk 的顶层 `task_metadata` 保存英文 `scene`、`task_description`、`objects`、源信息及指纹。帧内 `scene` 仍描述当前画面。翻译请求的用量单独记为 `task_metadata.translation_usage`，不计入逐帧 `usage`。

两个文件都没有时沿用原来的无先验流程；只有一个文件时使用可用信息，缺失的场景/任务翻译为 `unknown`。存在元数据却无法取得物体清单、YAML 格式错误或发现多个 `*_config.yaml` 时会明确报错，不会退回无约束识别。

断点续跑会复用翻译并校验上下文；从 raw 恢复时也会重新检查物体名称。任务文件、模型、提示词或采样配置变化，或者已有无上下文的旧结果时，须使用全新的输出目录，避免混用。旧结果不会自动删除。单视频模式可将 `--output_json` 指向新的目录，并确保自定义 raw、checkpoint、batch、chunk 路径也使用新目录；数据库模式使用固定输出路径，重跑前需自行归档旧 `annotation` 目录并将阶段状态重置为 `pending`。

## 从 L1 数据库批量标注

完成[数据库初始化与连接配置](../database/README.md)后，在项目根目录执行：

```bash
export DASHSCOPE_API_KEY="你的key"

python -m roboego_hand_vis.qwen_annotation.database_run \
  --shared-root ./data/l1 \
  --limit 3
```

`--shared-root` 使用下载时的共享根目录。省略 `--limit` 或设为 `0` 时处理所有当前可领取的数据。自定义配置使用 `parameters.api_key_env` 指定的环境变量提供 API key。

只领取下载、基础质检、视频粗质检均为 `completed`，且千问标注为 `pending` 的 L1 数据。不依赖立体矫正或手部采样检测，输入始终为原始 `videos/left.mkv`。

运行期间 `qwen_annotation_status=processing`，完整标注成功写 `completed`，执行异常、部分批次失败或输出不完整写 `failed`。不自动写入 `block` 或启动后续阶段。批次有失败时退出码为 `1`。

标注报告为 `annotation/qwen_frame_annotations_left.json`，抽帧、原始响应、checkpoint、batch 和 chunk 目录都放在各自 session 的 `annotation` 下。保留原来的断点续跑功能；失败任务不会自动领取重试，解决原因并确认旧进程停止后，将该阶段状态改回 `pending` 再运行。

数据库模式要求配置中的 `frames_dir`、`raw_output_dir`、`checkpoint_json`、`batch_output_dir`、`chunk_output_dir` 留空，由入口按 session 分配输出路径；暂不支持 `windows_json`，该项也需留空。原单视频模式仍支持自定义这些路径。
