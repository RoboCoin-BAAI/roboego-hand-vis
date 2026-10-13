# 本地运行，按阶段复制产出

全部 12 个阶段的 `database_run` 支持：

| 参数 | 作用 |
| --- | --- |
| `--shared-root` | 共享数据根目录，也是产出复制回去的位置 |
| `--local-root` | 可选，本地运行根目录；省略时直接在共享目录运行 |
| `--table` | 数据表，默认 `l1data` |

下载仍使用 `download_data.l1 --database --output-dir` 指定共享目录。
数据库里的 `local_path` 始终是 session 相对于共享根目录的路径，不因本地处理改变。

## 第一环节

```bash
python -m roboego_hand_vis.raw_headband_integrity.database_run \
  --shared-root /share/roboego/l1 \
  --local-root /local-cache/roboego/l1 \
  --limit 3 --overwrite
```

基础质检领取数据后，先检查本地 `session` 目录：存在就直接使用；不存在才从共享目录复制。
不需要额外复制参数。处理结束后，把该环节清单里的产出复制回共享目录，再更新数据库状态。
复制输入使用临时目录，复制完成才放到正式本地位置。

## 后续环节

```bash
python -m roboego_hand_vis.video_simple_quality.database_run \
  --shared-root /share/roboego/l1 \
  --local-root /local-cache/roboego/l1 \
  --limit 3
```

后续环节直接使用本地 session，不复制输入。也可以手动准备本地数据。
例如数据库 `local_path=session_xxx`，处理目录为 `/local-cache/roboego/l1/session_xxx`，
产出回传到 `/share/roboego/l1/session_xxx`。两个根目录不能相同或嵌套。

只指定 `--shared-root` 时，保留原来的单目录流程。
计算算法、模型配置、JSON/YAML 内容和其中记录的路径都保持原样。

## 各阶段回传清单

具体相对路径集中在 [stage_outputs.yaml](config/stage_outputs.yaml)，可按需求调整。
`paths` 保留环节原本的产出目录/文件作为说明；实际只复制 `required` 中列出的文件、目录或通配符。
`required: []` 表示不复制任何产出。同名文件覆盖，不删除共享目录中的其他文件。
阶段完成或门禁阻断时，每项都必须匹配到文件；执行失败时只回传已存在的选定文件。

| 阶段 | 原本产出（实际回传由 `required` 选择） |
| --- | --- |
| `raw_headband_integrity` | 质检报告、时间戳、可能修复了文件头的音频 |
| `video_simple_quality` | 视频粗质检 JSON |
| `detect_hand` | 左目手部检测 JSON |
| `stereo_rectification` | 左右矫正视频、标定、时间戳、meta、manifest 和日志 |
| `qwen_annotation` | 标注 JSON、checkpoint、chunk、采样图片、batch 和原始响应 |
| `post_annotation_actions` | 动作/复核/整理 chunk、合并动作、子任务结果和采样图片 |
| `stereo_depth_gate` | 深度前质检报告与采样可视化 |
| `fast_foundation_depth` | 逐帧深度、元数据、`vis` 预览、日志和运行配置 |
| `hand` | `hand_tracking.parquet`、`render.mp4`、运行日志和完成凭据 |
| `archive_depth_cache` | 归档视频、mask、预览、元数据、保留的 PNG 和日志 |
| `vio` | 轨迹、报告、manifest、整理后输入、估计器配置、统计和日志 |
| `video_privacy_blur` | 打码视频及其输出目录中的报告、日志和恢复状态 |

门禁报告同样回传，复制成功后写 `completed` 或 `block`；复制失败写 `failed`，保留本地产出。
各环节原有清理行为仍只作用于实际运行目录，复制操作不清理共享盘。
程序不自动删除本地 session，也不新增跨目录恢复、完成凭据或缓存来源管理。
