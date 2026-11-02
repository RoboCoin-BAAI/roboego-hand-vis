# L1 批量流水线

VIO 在启动数据处理前准备：先验证 ego Python 的 cv2/numpy/pyarrow/yaml，再安装 apt 依赖，统一配置并编译 `open_vins/ov_msckf/build-release`。使用 Unix Makefiles、`/usr/bin/make`、Release、ENABLE_ROS=OFF、CMAKE_POLICY_VERSION_MINIMUM=3.5；目标为 roboego_openvins_offline，产物 openvins_offline_native。L1 VIO compute.py 也必须使用 build-release 路径。构建后 ldd 检测缺失动态库，失败即停止启动；日志为 STATE_ROOT/vio_setup.*.log。不要配置 build-release 后又编译旧 build。

视频快速 QC 默认启用：新版转换器保存 `meta/l1_video_validation.json`；QC 核对视频大小、mtime_ns 和时间戳 parquet 的 SHA256 后，可复用已验证的 ffprobe 帧数/尺寸/FPS，继续以当前数据检查帧数匹配和其他模态。旧数据没有记录时仍解码一次，成功结果缓存到 `annotation/video_validation/`，后续可复用。移动目录保留文件大小和修改时间时不影响缓存；文件变化则失效。大小及时间戳不是视频内容哈希，若人为保留这两项替换视频内容，需要 `--full-video-check` 强制完整检查。质量报告的 decodable 检查中有 `reused_validation` 和 `validation_source`。该机制不跳过相机时间对齐、IMU、音频或标定检查；不自动生成未执行过的验证结果。

QC 标定补齐：流水线传入 `--repair-calibration-root "$OUTPUT_ROOT"` 后，先完成第一轮 QC，再从该根目录下已有 QC 通过记录、且 camera-camchain.yaml 结构完整的 session 建立供体快照。同 cp_编号可跨 batch/日期复制（用户确认同编号始终同设备同安装），优先时间最近的供体。仅替换缺失/结构不全的 camera-camchain.yaml；不复制相机—IMU 外参，不覆盖已完整的文件。原不完整文件保留为 `.before_repair-*`，来源及哈希记录在 annotation/camera_calibration_repair.json。修复后重新 QC，其他失败仍然有效；没有合格供体时 repair_structure 失败，不默认放行。不传该选项时保持现有的简单标定内容检查。同步 calibration_repair.py、compute.py、run.py 和流水线 sh 后生效。

校正现在默认校验并复用已有产物：比较当前标定推导的校正矩阵、balance、输入/输出裁剪、分辨率、内参、基线、时间戳、帧映射和数据集 feature；完整解码输出视频并核对帧数/FPS。通过时输出 `[skip-existing]`，不重写视频；搬目录只修复 manifest 路径。无效结果输出 `[recompute]` 并重算。单独运行校正入口可加 `--no-skip-existing` 强制重算。同步校正的 `compute.py`、`run.py` 和新增 `reuse.py` 后生效，无需修改启动命令。旧产物没有原始视频内容指纹，此校验不能证明同帧数的源视频内容从未被替换；如果主动更换过源视频，应强制重算。

入口是 `run.sh`，配置是同目录 `pipeline.conf`。只新增编排文件，不修改现有七个入口。适用于 Linux、Bash 4+，需要 `flock`、`setsid`、GNU coreutils。

## 依赖和资源

```text
CSV → 下载 → 转换 → 质检 → 立体校正 ┬→ 深度（8 GPU）
                                    ├→ VIO（CPU）
                                    └→ Argus（CPU + 远程 API）
```

下表来自本地代码检查；并发值是 128 核服务器的起始配置，没有服务器实测吞吐或峰值内存数据。

| 阶段 | 实际开销 / 输入输出 | 默认并发 |
|---|---|---|
| 下载 | 网络、磁盘写入；CSV → 原始 session。session 顺序下载，ks3util 内部固定 `--jobs=64` | 1 个 batch |
| 转换 | CPU 解码计数、时间戳/IMU处理、磁盘读写；视频 H264 → MP4 使用 `-c:v copy`，不是重新编码；输入输出均会 ffprobe 计数 | 8 session |
| 质检 | CPU、ffprobe、时间戳和传感器检查；写 `annotation/quality_report.json`；任一 session 不通过，入口返回非零 | 8 session |
| 校正 | CPU 解码、OpenCV remap、libx264 编码；每路编码显式 1 线程；新增双目视频、标定、时间戳并更新 `meta/info.json` | 12 session |
| 深度 | GPU 推理、CPU 双目解码、HEVC12 软件编码、磁盘；throughput 模式每卡一个常驻模型 worker，episode 动态队列 | 8 卡，每卡 CPU/decode 2、encode 2 |
| VIO | CPU OpenVINS、双目解码、IMU、轨迹质检；发布位姿及数据集元信息；生成配置中 OpenCV 4 线程，初始化最大 6 线程 | 6 session |
| Argus | 校正左目 MP4 + annotation 配置；本地视频准备和质检，远程 Qwen 推理；不消费本流水线生成的深度/位姿 | 4 视频，每视频 jobs=1、API concurrency=1 |

VIO 的 6 个 worker 并不等于 6 个 CPU 线程。OMP/MKL/OpenBLAS/OpenCV 环境线程数设为 1，以限制其他库的线程膨胀；OpenVINS 自身配置的线程数仍然有效。解码库也可能有额外线程，因此这些不是 CPU 硬配额。

深度输出在独立派生目录，不修改 `meta/info.json`；VIO 更新数据集元信息；Argus 使用独立 annotation/argus 工作目录，输入是单个 MP4。三者当前可同时运行。校正与 VIO 同时写同一 session 则不安全，所以校正是严格前置条件。若以后将 Argus 改成读取深度或位姿，必须重新调整依赖。

## 运行

把新增的整个 `scripts` 目录同步到服务器仓库，检查配置中的绝对路径。原始数据目录是 `L1`，输出是命令里使用的 `L1_lerobot`（下划线）。确认环境已有各项目依赖、OpenVINS 已构建、模型权重和 ks3 配置可用。

```bash
cd /share/project/liuyou/humancoin/code/roboego-hand-vis
bash -n scripts/old/L1/run.sh
# 通过服务器现有秘密管理方式设置 DASHSCOPE_API_KEY，不要写入配置或日志。
bash scripts/old/L1/run.sh batch_0000001

# 不带 batch 参数，按文件名顺序处理 INPUT_ROOT 下所有 batch_*.csv。
mkdir -p /share/project/liuyou/humancoin/data/L1_pipeline_state
nohup bash scripts/old/L1/run.sh \
  > /share/project/liuyou/humancoin/data/L1_pipeline_state/pipeline.log 2>&1 &
echo $!
```

可以传多个 batch 名称指定处理顺序。首先跑一个 batch，观察各阶段日志中的时间、`nvidia-smi`、`pidstat` 和 `iostat -xz 5`。先以默认并发运行，GPU 等待 CPU/磁盘时不要继续提高 CPU worker 数。若校正明显供不上深度，且 CPU/存储还有余量，可把校正从 12 调至 16；共享存储出现高延迟时先降转换/校正并发。1T 内存并不意味着要把视频全部载入内存，当前入口主要流式处理。

7 个阶段各有一条队列，每个阶段同一时刻只处理一个 batch。A 的深度/VIO/标注运行时，可以准备 B；`MAX_INFLIGHT=2` 保证 C 的下载等到 A 的三个末端阶段均结束。失败也视作结束，从而让后续 batch 继续。

窗口限制在途工作量，不限制累计磁盘占用；原始、转换后、校正、深度结果全部保留，不自动删除。开始前按首个 batch 实测磁盘增量估算整批空间。8 卡模型实际显存需要实测，A100 40GB/80GB、输入分辨率和模型配置都会影响结果。

## 失败、续跑与预算

- 任一阶段失败只阻断该 batch 的依赖任务；例如 VIO 失败不影响深度和标注，QC 失败会阻断整个 batch 的校正及后续阶段。当前采用严格整 batch 质检门禁，不自动丢弃坏 session。
- 默认非 API 阶段最多执行两次；标注不自动重试，以免反复发付费请求。修复后再次运行相同命令即可；标注的已付费进度由 Argus 自身 receipts/budget 管理。
- 状态路径是 `STATE_ROOT/batch名/配置与CSV哈希/`。每阶段保存实际命令、逐次日志、exit code 和 `.done/.failed/.blocked` 标记。正常退出码为 0，有失败/阻断为 1。输出目录使用 flock 防止两个编排实例同时运行。
- `.done` 表示该入口曾返回 0，并不逐次重新校验产物。不要手动修改/删除已完成输出后直接相信标记；配置、脚本或 CSV 改变会启用新状态。其他 Python 源码/权重或环境变量的改变不在哈希内，需要手动选择新的 `STATE_ROOT`。入口可能有自己的跳过策略，例如转换按 provenance 跳过已有 session；切换状态目录不等于强制重新转换。
- 想重跑某个已成功阶段，应停止编排后删除它和全部后继阶段的 `.done`，再重启。不要在旧任务仍运行时移除 Argus lock；异常断电可能留下 `.argus_batch.lock`，需先确认对应进程已不存在再手动处理。
- Ctrl+C 或 `kill -TERM 主进程PID` 会停止队列及其活动子进程组；强制 kill -9/断电不能保证清理，不会写入成功标记。重试按阶段进行：校正、深度、VIO 可能重算整个 batch，而非只重算失败 session；转换自身支持 session 跳过。
- `REVIEW_CAP_CNY=0.6` 沿用记录的命令，是每个视频的估算预算，非整个 batch 或全部数据的总预算，也不是服务端硬限额。大约 N 个视频对应 N×0.6 元的配置估算上限，实际以服务端计费为准；预算不足可能导致标注未完成。`--cap 0` 是该 CNY 预算模式要求的参数。
- VIO 默认保留原命令的 `--zupt`。可将 `VIO_POLICY` 改为 `--zupt-fallback`，先普通 VIO，再在失败/质量不通过时尝试 ZUPT；这会改变估计策略，不是单纯性能调优。

本机 Windows 没有服务器的模型、数据访问和 GPU 环境；不会在本机执行真实下载、推理或收费标注。

验证：通过 Git Bash 的 `bash -n`；用两个模拟 batch 验证了 QC 失败阻断、独立 batch 继续运行、深度失败自动重试、窗口为 1 的流水运行、成功阶段在第二次执行时跳过，以及最终非零退出码。模拟中替代了 Linux 的 flock/setsid，因此真实文件锁及进程组终止行为仍需在服务器验证。

## ?????

?????? Job?????? scripts ???roboego_hand_vis/lerobot_headband_integrity/L1/run.py?lerobot_stereo_rectification/L1/run.py?lerobot_fast_foundation_depth/L1/run.py?lerobot_vio/L1/run.py??? argus/review/__main__.py ? batch.py?????? sh??????????? --sessions-file?

????????? L1/run.py?????? argus/review/L1/run.py?????????? L1/run.py?????? argus/review/L1/run.py????? session ???????????/??/VIO ??????????

????????? L1/run.py?????? argus/review/L1/run.py??????????QC ??????? `--repair-calibration-root "$OUTPUT_ROOT"` ???????? QC????????? QC ???????? VIO ??????? session ???????? cp_????? batch/??????????????? `camera_imu-imu.yaml` ? `camera_imu-camchain-imucam.yaml` ???/????????? L1/run.py?????? argus/review/L1/run.py??????????? `camera-camchain.yaml`????????????/????? annotation/camera_calibration_repair.json?????? QC???????? VIO ??????????????
