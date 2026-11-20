---
license: cc-by-nc-4.0
gated: true
extra_gated_prompt: "Please provide your name, email address and intended use to request access to HumanCOIN datasets."
extra_gated_fields:
  Name: text
  Email: text
  Intended use: text
viewer: false
language: [en, zh]
pretty_name: "l1_living_room_cover_task_1920x1456_30fps_2views"
tags: ["HumanCOIN", "L1", "egocentric", "multi-sensor", "cover_task"]
---

# l1_living_room_cover_task_1920x1456_30fps_2views

## Dataset description

Read with [HumanCOIN](https://github.com/FlagOpen/HumanCOIN).

```bash
pip install -U "humancoin>=0.1.1"
```

This L1 dataset groups independent task recordings under one repository. Each `unique_task_name/` directory is a complete HumanCOIN recording with its own metadata and master timeline.

**Example task description (first task):** the person unfolds the sofa cover and puts it on the sofa to complete the tidying.

| Property | Value |
| --- | --- |
| Repository ID | HumanCOIN/l1_living_room_cover_task_1920x1456_30fps_2views |
| Data tier | L1 |
| Task category | cover_task |
| Task labels | cover sofa |
| Scene | Indoor / Household / Living room |
| Hand capture | bare_hand |
| Tasks | 1 |
| Master frames | 4297 |
| Master frame rate (Hz) | 30.0 |
| Duration | 143.23 s (0.040 h) |
| Data format | HumanCOIN-LeRobot |
| Files (excluding README.md) | 30 |
| Size (excluding README.md) | 267.77 MiB |

Task count is the number of `unique_task_name/` recordings, including recordings with the same task label. Frame counts, durations and file statistics are summed across all tasks. File statistics include each task's `meta/readme.json` and exclude this dataset card.

Scene, tier, example description and shared stream/storage metadata below come from the first task in directory-name order.

## Actions and objects

**Atomic actions (deduplicated union):** adjust, pick, pull, unfold

| Object |
| --- |
| sofa |
| sofa cover |

## Sensor streams

Shared feature definitions from the first task. Video shapes use HWC order; loaded images use CHW. Independent sensor rates and query windows are preserved. Per-stream frame/sample counts are not aggregated here.

| Feature | Type | Shape | Rate (Hz) | Query window |
| --- | --- | --- | --- | --- |
| observation.images.head_left_rectified | video | [1456, 1920, 3] | 30 | nearest |
| observation.images.head_right_rectified | video | [1456, 1920, 3] | 30 | nearest |
| observation.body_pose | float32 | [154] | 30 | nearest |
| observation.head_pose | float32 | [7] | 30 | nearest |
| observation.head_camera_pose | float32 | [7] | 30 | nearest |
| observation.hand_left_points | float32 | [63] | 30 | nearest |
| observation.hand_left_quaternions | float32 | [84] | 30 | nearest |
| observation.hand_right_points | float32 | [63] | 30 | nearest |
| observation.hand_right_quaternions | float32 | [84] | 30 | nearest |
| observation.right_eye_camera_pose | float32 | [7] | 30 | nearest |
| observation.upper_body_pose | float32 | [98] | 30 | nearest |
| observation.lower_body_pose | float32 | [56] | 30 | nearest |
| eye_gaze_predict | float32 | [2] | 30 | nearest |
| observation.images.depth_head_left_rectified | depth video | [1456, 1920, 1] | 30 | nearest |

**Primary camera:** observation.images.head_left_rectified

## Storage and time alignment

Each task has the same relative layout; timeline indices restart inside each task:

```text
l1_living_room_cover_task_1920x1456_30fps_2views/
├── README.md
└── <unique_task_name>/
    ├── annotation/
    ├── calibrations/
    ├── data/
    ├── meta/
    └── videos/
```

| Component | Path template |
| --- | --- |
| index_path | <unique_task_name>/data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}/master_index.parquet |
| data_path | <unique_task_name>/data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}/{feature_key}.parquet |
| video_path | <unique_task_name>/videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4 |

Reading `ds[i]` queries each feature using that task's master-index timestamp. A query window selects samples in `(t + start, t + end]`; `nearest` selects the closest sample. Task labels are stored in each task's `meta/tasks.jsonl`.

## Loading

Load a task directory inside the downloaded repository:

```python
from pathlib import Path
from humancoin import HumanCOINDataset

repo_id = "HumanCOIN/l1_living_room_cover_task_1920x1456_30fps_2views"
dataset_root = Path("data") / repo_id
task_dirs = sorted(p for p in dataset_root.iterdir() if (p / "meta/info.json").is_file())

for task_dir in task_dirs:
    ds = HumanCOINDataset(repo_id=repo_id, root=task_dir, video_backend="pyav")
    item = ds[0]
    print(task_dir.name, len(ds), item.keys())
```

Depth videos are read as native single-channel values without RGB conversion or normalization. Depth units, scale and encoding are declared in each task's `meta/info.json`. Supplementary masks and depth metadata, when provided, are stored in that task's annotation directory.

## Annotations and calibration

Shared artifact paths, taken from the first task:

- `<unique_task_name>/annotation/episode_000000_structured.json`
- `<unique_task_name>/calibrations/dataset_manifest.json`
- `<unique_task_name>/calibrations/rectify_params.yaml`
- `<unique_task_name>/calibrations/source_camera_calibration.json`

## Authors and contact

Shihan Wu, You Liu, Chunyu He, Minghui Qin, Xiang Yang, Longhan Jiao, Shaoxuan Xie, Runtian Xu, Tengfei Pan, Yance Jiao, Yunyan Zhang, Kaiqian Tian, Wenjie Chen, Jingrui Pang, Zhiwei Yu, Yihang Li, Jiangmiao Pang, Kai Zhu, Jianxing Chen, Sihan Liu, Xiaofeng Wang, Zheng Zhu, Jie Zhao, Jing Li, Zhongpu Xia, Jianwei Niu, Wei Li, Zhexuan Yang, Yue Cao, Ji Zhao, Dongsheng Jiang, Yongwei Wang, Yunge Liu, Bo Lei, Huan Yang, Tiejun Huang, Zhongyuan Wang, Guocai Yao

**Institution:** Beijing Academy of Artificial Intelligence

**Contact:** humancoin@baai.ac.cn

**Project:** https://koorye.github.io/HumanCOIN-dev/

## License

This dataset and its documentation are released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). This dataset card uses the license selected for this release.

## Citation

```bibtex
@techreport{wu2026humancoin,
  title = {HumanCOIN: The Human-Signal Dataset at the Scale of Egocentric Manipulation},
  author = {Shihan Wu and You Liu and Chunyu He and Minghui Qin and Xiang Yang and Longhan Jiao and Shaoxuan Xie and Runtian Xu and Tengfei Pan and Yance Jiao and Yunyan Zhang and Kaiqian Tian and Wenjie Chen and Jingrui Pang and Zhiwei Yu and Yihang Li and Jiangmiao Pang and Kai Zhu and Jianxing Chen and Sihan Liu and Xiaofeng Wang and Zheng Zhu and Jie Zhao and Jing Li and Zhongpu Xia and Jianwei Niu and Wei Li and Zhexuan Yang and Yue Cao and Ji Zhao and Dongsheng Jiang and Yongwei Wang and Yunge Liu and Bo Lei and Huan Yang and Tiejun Huang and Zhongyuan Wang and Guocai Yao},
  institution = {Beijing Academy of Artificial Intelligence},
  year = {2026},
  url = {https://koorye.github.io/HumanCOIN-dev/}
}
```
