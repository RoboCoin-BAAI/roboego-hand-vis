# Qwen 3.7 批量视觉场景分类

入口：[reclassify_scenes_qwen.py](reclassify_scenes_qwen.py)。Python 3.10+，使用标准库、FFmpeg/FFprobe，以及 Argus 仓库中的共享网关配置。默认模型 `qwen3.7-plus`。

对每个 `batch_*/<样本名>/annotation/episode_*_structured.json`：

1. 找到对应 episode 的左目 RGB 视频。
2. 用 FFprobe 读取帧数、帧率和时长元数据，不使用 `-count_frames` 或逐帧扫描。
3. 在约 10%、50%、90% 的位置分别抽一张 JPEG，默认缩放至宽 960 像素。
4. 将三张图片、`task_description`、`scene_classification.original_description` 一起发送给模型。
5. 按本地 V2 词表严格校验结果，保存分类依据；指定 `--apply` 时只写回 `category` 和 `area`。

## 分类限制

机器词表：[scene_taxonomy_v2.json](scene_taxonomy_v2.json)，与目前 V2 文档对应，包含 6 个一级分类、38 个已有二级分类。此前的 `massage_room` 已按用户要求归入 `professional_work / conference_center`，不再生成该临时分类。

- 一级分类固定，禁止新增。
- 只有 `commercial_service` 可以新增二级分类，而且优先使用已有枚举；其他一级分类的二级分类必须出现在词表中。
- 已有二级分类不允许跨大类使用；新增名称必须为长度不超过 64 的英文 `snake_case`。
- 会议室归 `professional_work / conference_center`；室内体育及羽毛球场景归 `public_space / indoor_sports_venue`。
- 不再接受已经归并的 `meeting_room`、`badminton_court`、`food_service_area` 等临时分类。
- 无法确定时模型应返回 `other / other`。非法模型结果、视频错误、网络错误会记入错误清单，不会被伪装成 other，也不修改原文件。
- 默认不将所有样本预设为家庭场景。确有场所背景时，可通过 `--context` 提供，限本次选中的样本。

## 视频选择与抽帧

自动依次寻找以下视频目录，选择第一个匹配当前 episode 的视频：

```text
videos/chunk-*/observation.images.head_left_rectified/episode_*.mp4
videos/chunk-*/observation.images.headband_left_rectified/episode_*.mp4
videos/chunk-*/observation.images.head_left/episode_*.mp4
videos/chunk-*/observation.images.headband_left/episode_*.mp4
```

可以用 `--video-key observation.images.<实际RGB特征名>` 指定其他目录。默认不会读取深度视频。同一 episode 在同一特征下匹配多个 chunk 时会报错，避免误选视频。样本目录名不限制为哈希，支持其他数据集的样本名及多个 episode。

有 `nb_frames` 和帧率时，目标索引为 `min(N-1, floor(N * 比例))`，再换算为 seek 时间；缺少帧数元数据时，使用时长的 10%、50%、90%。输入前使用 `ffmpeg -ss ... -i ... -frames:v 1`，从邻近关键帧开始局部解码，避免从视频开头做全量 select。可变帧率视频的位置是近似值；长 GOP 视频仍可能需要解码较长的关键帧区间。

## 服务端文件与凭据

本次工具已同步到：

```text
/share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools/
├── reclassify_scenes_qwen.py
├── apply_scene_classification.py
└── scene_taxonomy_v2.json
```

复制到别处使用时，三个文件需要放在同一目录。`--argus-root` 指向 Argus 仓库根目录。本机两个仓库并列放置时可自动发现 Argus。

凭据直接复用 `argus/review/qwen_endpoint.py`，不在脚本中保存密钥：配置了 `XMSXB_API_KEY` 时走 XMSXB（可选 `XMSXB_BASE_URL`）；否则使用 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。端点拼接及请求差异沿用共享代码。服务端模型权限、视觉支持和实际额度需要账户配置，本次没有实际调用模型 API。

服务器使用项目环境 `/share/project/liuyou/humancoin/envs/ego/bin/python`，系统默认 `python3` 版本较旧，不能用于本脚本。

## 运行

以下示例对其他数据根目录 `/path/to/dataset` 运行。它下面应包含 `batch_0000001` 等目录；报告目录放在数据根目录之外。

先抽一个样本预览（无模型请求，不修改标注）：

```bash
export PATH=/share/project/liuyou/humancoin/envs/ego/bin:$PATH

python /share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools/reclassify_scenes_qwen.py \
  --root /path/to/dataset \
  --start 1 --end 10 --limit 1 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --report-dir /path/to/scene_classification_reports \
  --dry-run
```

批量分类并直接覆盖两个分类字段：

```bash
python /share/project/liuyou/humancoin/data/guanglun_scene_qwen_tools/reclassify_scenes_qwen.py \
  --root /path/to/dataset \
  --start 1 --end 10 \
  --argus-root /share/project/liuyou/humancoin/code/argus \
  --report-dir /path/to/scene_classification_reports \
  --model qwen3.7-plus --workers 4 \
  --apply
```

如果只想调用模型生成分类报告，去掉 `--apply`。之后使用相同参数和报告目录加上 `--apply`，会复用已完成的模型结果进行写回，不重复请求。

| 参数 | 用途 |
|---|---|
| `--start 40 --end 60` | 批次范围，含首尾；目录格式为 `batch_0000040` 等 |
| `--batches batch_0000040 batch_0000042` | 指定不连续的批次，与 start/end 二选一 |
| 不指定批次参数 | 处理数据根目录下全部 batch 目录 |
| `--only-other` | 仅处理当前 category 为 other 的标注 |
| `--limit 10` | 仅处理前 10 份标注 |
| `--workers 4` | 并发样本数，默认 1 |
| `--context '本次选中样本均拍摄于家庭环境'` | 为已确认的范围提供场所背景，不应对未知数据盲目添加 |
| `--ffmpeg` / `--ffprobe` | 可显式指定可执行文件路径 |
| `--timeout 180` | 单次模型请求超时秒数 |
| `--media-timeout 60` | 单次探测或抽帧超时秒数 |
| `--max-tokens 4096` | 模型输出 token 上限 |
| `--retry-failed` | 明确允许对上次失败或结果未知的请求再次调用，可能产生额外费用 |

## 输出、恢复与写回保护

- `summary.json`：本次选择范围的成功、失败、other、实际写回数量等。
- `results.jsonl`：每份标注的处理结果、旧分类、新分类及依据。
- `other_samples.json` / `other_samples.csv`：本次结果中的 other 样本。
- `errors.json`：本次失败明细。
- `new_commercial_areas.json`：本次新增的商业服务二级分类。
- `jobs/`：各样本的抽帧、输入证据、原始模型响应、缓存结果和写回前原文件副本。

每次运行会重写上述本次汇总与清单，但保留 `jobs/` 中的历史缓存、请求回执和备份。相同输入描述、视频属性、模型、提示词及采样参数下复用缓存。源视频或描述发生变化时会重新分类；代码或词表改变也会产生新的缓存身份。

API 请求发出前记录回执，失败或超时不会自动重试。重新运行会继续其他样本，但对可能已计费且没有可恢复响应的请求报错；确认后可使用 `--retry-failed`。恢复得到的缓存响应仍须通过严格分类校验。

写回前检查视频和完整标注未被并发修改，备份保存于外部报告目录，使用原子替换，只替换两个已有字段的 JSON 值并核对结果。保留其他字段、描述原文及 JSON 排版，不往原文件中插入依据、状态或其他字段。原标注缺少这两个字段时会报错，不自动新增。

## 验证

8 项本地测试通过，覆盖分类边界、输入 seek、视频目录选择、多模态请求、缓存复用、超时恢复及字节级写回约束。服务器一个真实光轮视频的三点抽帧预检通过，未调用模型 API，未修改标注。

```bash
python -m unittest discover -s tests/readme_upload -p test_scene_reclassification.py -v
```
