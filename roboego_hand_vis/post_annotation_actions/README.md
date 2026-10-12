# 标注后字段整理

读取千问标注 chunk，整理规范动作字段并合并结果；默认还会生成子任务。此阶段调用配置的千问服务，不作为门禁。文本整理、视觉对象整理和子任务生成默认统一使用 `qwen3.8-27b`，可通过配置中的模型字段修改。

## 处理指定数据

```bash
python -m roboego_hand_vis.post_annotation_actions.run \
  --qwen_chunk_dir /path/to/session/annotation/qwen_frame_annotation_chunks \
  --output_dir /path/to/session/annotation \
  --overwrite
```

参数见 `recommended.yaml`，可通过 `--config` 指定其他配置。此命令不回写数据库。

`parameters.base_url` 是完整的模型接口地址，默认使用业务空间专属地址：`https://llm-g3o8d3j71xbf6prc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions`。文本整理、视觉对象整理和子任务生成均使用这个地址。留空时优先读取 `DASHSCOPE_BASE_URL`，未设置则使用上述默认地址。

## 从 L1 数据库批量处理

完成[数据库初始化与连接配置](../database/README.md)后，在项目根目录执行：

```bash
export DASHSCOPE_API_KEY="你的key"

python -m roboego_hand_vis.post_annotation_actions.database_run \
  --shared-root ./data/l1 \
  --limit 3 \
  --overwrite
```

`--shared-root` 使用下载时的共享根目录。省略 `--limit` 或设为 `0` 时处理所有当前可领取的数据；`--overwrite` 覆盖已有整理 chunk，省略时按原配置复用已有结果。

只领取下载、前两项质检和千问标注均为 `completed`，且字段整理为 `pending` 的 L1 数据。不依赖立体矫正或手部采样检测。

运行期间 `post_annotation_actions_status=processing`，成功且结果齐全写 `completed`；缺少 chunk、输出不完整或执行异常写 `failed`。不自动写入 `block` 或启动后续阶段，批次有失败时退出码为 `1`。

输入为各 session 的 `annotation/qwen_frame_annotation_chunks`。输出均放在该 session 的 `annotation` 下：

- 整理 chunk：`stage1_action_resolved_chunks`。
- 合并动作：`stage1_actions_left_v1_resolved.json`。
- 子任务：`stage1_subtasks_left_v1.json`（启用 `generate_subtasks` 时）。

数据库模式要求自定义输出目录和 `frames_dir` 配置留空，由入口按 session 分配。模型、API key 环境变量、子任务和视觉对象整理开关仍使用原配置。

某条任务执行失败后不会自动重试。确认接口配置正确后，可在数据库工具中仅将该记录的 `post_annotation_actions_status` 从 `failed` 改回 `pending`，再运行批处理命令；已有整理结果默认复用。

接口报错会保留 HTTP 状态、调用阶段（`text_resolver`、`vl_object_resolver`、`subtask_resolver` 或 `subtask_object_resolver`）、模型名、请求地址和服务器错误正文，便于区分模型不可用与接口路径错误。
