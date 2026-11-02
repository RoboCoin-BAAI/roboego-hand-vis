# ego_hands_pose 的 LeRobot 适配

模型目录为相邻仓库 ego_hands_pose/video_reconstruction，使用 infer.py。
L1/run.py 加载 head_left_rectified，L3/run.py 加载 headband_left_rgb_rectified。
仅处理左目，不把左右目合并推理。流水线 hand 阶段已启用，依赖 QC 和去畸变通过。

```bash
bash scripts/L1/pipeline_gpu.sh --batches batch_0000001 --gpus 0,1,2,3 --watch
# 只重跑手部
bash scripts/L1/pipeline_gpu.sh --batches batch_0000001 \
  --run-stages hand --force-stages hand --gpus 0,1
```

YAML 的 hand_model_root 指向 video_reconstruction；hand_python 指向模型运行环境。
模型要求 Python 3.11+、CUDA PyTorch 和其 requirements.txt，不会自动安装。
传入 GPU ID 后子模型只看到一张卡，内部使用 cuda:0。
与深度共用设备槽，每卡同时只运行一个任务；两种任务可分布在不同卡上。
模型进程按 episode 启动，尚未实现跨任务常驻权重。

默认完整视频、precision=auto、window_batch_size=1、long_edge_pad 和 --no-overlay。
长边缩放补边保留宽屏完整画面，不默认中心裁掉左右手。
输出 hand/chunk-XXX/episode_XXXXXX/：prediction.npz、metadata.json、source_timestamps.parquet、lerobot_hand.json 和 model.log。
原始 LeRobot 时间戳原样保留；模型 NPZ 内的解码时间戳不替代采集时间戳。
检查预测帧序/数量与 Parquet 记录一致，并核对模型提供的预测 SHA256。
不自动注册 LeRobot 手部特征，不覆盖 L3 原始手套观测。

重要：模型 camera-json 仅用于叠加显示，不改变三维预测，本适配不传入它。
三维结果属于模型估计的相机坐标，不保证与标定左右目坐标、尺度或 VIO 已对齐。
不要直接用去畸变标定内参、基线将这些预测当成几何真值。
模型/权重更新后需显式 force；复用依据为输入视频大小/mtime、时间戳摘要和运行设置。

新结果在隐藏临时目录生成，通过校验后发布；模型失败保留日志及旧产物。
本地只执行 CPU 适配测试，未运行真实 GPU 模型。

独立命令现在将扫描状态、每个 session 检查、模型 PID、加载/推理输出及 traceback 实时打印到 Job。
同样内容追加保存到默认 data/pipeline_tasks/L1/hand_standalone/console.log
（按 --source 的父目录推导，可用 --log-dir 显式指定）。启动时立即打印统一日志位置，
扫描/任务执行期间每 30 秒输出存活提示，子模型无输出时每 20 秒提示运行中。
每任务 model.log 仍保留原始子模型输出；模型 Python 使用无缓冲输出。

--vio-success-only 默认从数据根目录旁的 L1_pipeline_state/batch_*/运行目录/vio.*.log
提取候选 session，逐 batch 打印读取进度，不再递归查找数据目录的 meta/info.json。
--pipeline-state-root 可显式指定旧状态根目录；候选仍须通过当前发布 VIO 报告的复核。
--sessions-file 则直接使用指定清单。不使用 VIO 筛选时只列 batch/session 两级目录。

## ??????????

????????? VIO ??????????????????
`data/pipeline_tasks/L1/hand_standalone/requests/<??ID>/selected.json`?
?? `[hand-selected]` ???????? `--select-only` ????????? GPU ???
????????????? worker??? worker ???? Predictor???????
???????????????? gpu-N.json?gpu-N.log?gpu-N.results.jsonl?
????????? VIO ????????????????
? per-episode model.log ??????????????
GPU ????? session ??????????????????
???? batch ?????????????????
????????? Predictor ?????? session??? GPU ??????????
