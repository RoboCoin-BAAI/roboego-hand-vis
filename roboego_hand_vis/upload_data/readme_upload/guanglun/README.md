# 光轮正式上传文件清单

**物品名称与动作词表推荐使用统一入口**：[COMBINED_TERMS_CLEANING.md](COMBINED_TERMS_CLEANING.md)。先生成一个独立的替换结果 JSON，不改原标注；等场景清洗完成后再统一应用，只替换指定字段并保留最新场景。

动作词表清洗见 [ACTION_VOCABULARY_CLEANING.md](ACTION_VOCABULARY_CLEANING.md)：参考 Subtask/Atomic 词表，仅处理 `actions.vocabulary`，规范基础动词并合并同义项。

光轮与简智的物体名称清洗见 [OBJECT_NAME_CLEANING.md](OBJECT_NAME_CLEANING.md)：参考 HumanCOIN_Object.xlsx，仅处理 `objects[].name`，使用 Qwen 3.7 去除量词及修饰。

批量视觉场景分类入口见 [SCENE_QWEN_BATCH.md](SCENE_QWEN_BATCH.md)：Qwen 3.7、10%/50%/90% 三点抽帧、严格 V2 分类，仅商业服务可新增二级分类。

场景清洗的分类规则、统计及新增二级分类见 [SCENE_CLEANING_01_39.md](SCENE_CLEANING_01_39.md)；待处理样本见 [scene_other_samples.csv](scene_other_samples.csv)。

当前剩余 34 个 other 样本的任务分组见 [OTHER_TASK_GROUPS.md](OTHER_TASK_GROUPS.md)，可筛选明细见 [scene_other_grouped.csv](scene_other_grouped.csv)。原 950 个 other 样本的家庭复分结果见 [HOUSEHOLD_OTHER_REVIEW.md](HOUSEHOLD_OTHER_REVIEW.md) 和 [scene_household_review.csv](scene_household_review.csv)。

本清单记录已确认的光轮数据上传范围。每个样本保留以下相对路径，共 **28 个文件**：标注 1 个、标定 2 个、Parquet 数据 18 个、元数据 4 个、视频 3 个。

服务器数据根目录：`/share/project/liuyou/humancoin/data/guanglun_depth`。目录层级为 `<batch>/<sample_id>/`，下面的路径均相对于样本根目录。

确认样本：`batch_0000001/018b12e72adf9160e7ddc4c95cf7e45976fbf09d5ed280cb4bacba0e90c7b25a`。

## 必需文件

```text
<sample_id>/
├── annotation/
│   └── episode_000000_structured.json
├── calibrations/
│   ├── dataset_manifest.json
│   └── rectify_params.yaml
├── data/
│   └── chunk-000/
│       └── episode_000000/
│           ├── eye_gaze_predict.parquet
│           ├── master_index.parquet
│           ├── observation.bad_frame_pose_hand.parquet
│           ├── observation.bad_frame_pose_lower_body.parquet
│           ├── observation.bad_frame_pose_upper_body.parquet
│           ├── observation.body_pose.parquet
│           ├── observation.hand_left_points.parquet
│           ├── observation.hand_left_quaternions.parquet
│           ├── observation.hand_right_points.parquet
│           ├── observation.hand_right_quaternions.parquet
│           ├── observation.head_camera_pose.parquet
│           ├── observation.head_pose.parquet
│           ├── observation.images.depth_head_left_rectified.parquet
│           ├── observation.images.head_left_rectified.parquet
│           ├── observation.images.head_right_rectified.parquet
│           ├── observation.lower_body_pose.parquet
│           ├── observation.right_eye_camera_pose.parquet
│           └── observation.upper_body_pose.parquet
├── meta/
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── info.json
│   └── tasks.jsonl
└── videos/
    └── chunk-000/
        ├── observation.images.depth_head_left_rectified/
        │   └── episode_000000.mp4
        ├── observation.images.head_left_rectified/
        │   └── episode_000000.mp4
        └── observation.images.head_right_rectified/
            └── episode_000000.mp4
```

`observation.upper_body_pose.parquet` 是必需的数据文件，非空目录。

## 不上传的文件及目录

- `calibrations/source_camera_calibration.json`
- 样本根目录及各级子目录中的 `logs/`、日志文件（例如 `ks3util.log`）
- `ks3util_output/`
- 锁文件（例如 `..reencode-chunk-000-episode_000000.publish.lock`）
- 其他未列入上述必需文件清单的文件及空目录

本清单仅记录正式数据文件范围；本次未修改上传脚本，未整理或删除服务器文件，也未执行上传。后续整理和上传应以此清单为准，缺少深度等必需文件的样本不能视为结构完整。

## 检查固定文件是否存在

在服务器运行（Python 3，无额外依赖）：

```bash
python3 check_required_files.py > missing_files.txt
# 自定义数据根目录和 batch 范围：
python3 check_required_files.py --root /share/project/liuyou/humancoin/data/guanglun_depth --start 1 --end 39 > missing_files.txt
```

默认检查前 39 个 batch 下所有 64 位十六进制样本 ID 目录。标准输出逐行打印缺失文件的完整路径；进度、汇总及读取错误输出到标准错误。只检查是否为文件，不检查大小、内容或帧数。退出码：0 全部存在，1 有缺失，2 有检查错误（含 batch 不存在或没有样本）。脚本只读，不修改数据。
