# OpenVINS 头部轨迹估计

沿用原 OpenVINS 离线准备与执行流程，不作为质量门禁。

## 处理指定数据

```bash
python -m roboego_hand_vis.vio.run \
  --session /path/to/session \
  --input_mode rectified
```

此命令保持原流程，不回写数据库。

## 从 L1 数据库批量处理

完成[数据库连接配置](../database/README.md)，安装当前 Python 环境的项目 `database` 依赖，并按本阶段 `recommended.yaml` 配置 OpenVINS 二进制和准备脚本使用的 Python 后，在项目根目录执行：

```bash
python -m roboego_hand_vis.vio.database_run \
  --shared-root ./data/l1 \
  --limit 3 \
  --input_mode rectified
```

省略 `--limit` 或设为 `0` 时处理当前所有可领取 session。可用 `--config` 指定其他 VIO YAML。数据库模式固定使用 `rectified`，省略 `--input_mode` 也一样；原手动入口的 raw 模式保留。

### 通过 chroot 脚本运行

如果 OpenVINS 和准备阶段的 Python 依赖仅安装在 rootfs 中，增加 `--chroot-runner`：

```bash
python -m roboego_hand_vis.vio.database_run \
  --shared-root /share/project/liuyou/data/EGOL1/data \
  --local-root /home/humancoindata/EGOL1 \
  --limit 3 \
  --input_mode rectified \
  --chroot-runner /share/project/liuyou/vio-run.sh
```

该模式跳过宿主机的 `runtime.system_python` 和 `openvins.binary` 文件检查，由脚本负责进入 chroot、使用其中的环境和代码。宿主机仍需能运行数据库入口及其依赖。省略参数时保持原来的宿主机执行方式。

脚本接口为 `bash SCRIPT SESSION EMPTY_OUTPUT run --input_mode rectified [参数...]`，须返回实际运行退出码，并将本次结果复制到 `EMPTY_OUTPUT`。还会传入 YAML 中的 `--start_s`、`--duration_s`（设置时）、`--stereo_tolerance_s`、`--max_imu_gap_s`、`--max_imu_boundary_gap_s`，以及禁用下采样时的 `--no-downsample_cameras`；rootfs 内的 `vio-runtime` 必须支持这些选项。宿主机 YAML 的环境路径不会覆盖 rootfs 内部配置。

每次 chroot 运行使用 session 内新的 `.vio-chroot-*` 空目录。脚本须复制出 `report.json` 和 `trajectory.npz`，使用 `roboego.vio.delivery.v2` 报告和 `openvins.frame_aligned.v2` NPZ。报告的 `input.session` 必须与当前 session 一致，`artifacts.trajectory` 为相对文件名 `trajectory.npz`。校验通过后才替换 `session/vio`，不再要求导出内部 `run_manifest.json` 或 JSONL。

原 `vio` 保留为 `.vio-previous-*`；失败时保留临时结果及原 `vio`，数据库写入 `failed`。结果按本地工作区规则同步回共享盘，无需手动写库。

宿主机需要 `flock`，且脚本所在目录需允许创建 `SCRIPT.database.lock`。使用同一个脚本绝对路径的数据库任务会串行执行，避免共用 rootfs 的 `/data/output` 冲突；手动运行脚本或使用其他脚本访问同一 rootfs 时，也必须协调使用该锁。

`--chroot-runner` 不改变脚本的数据映射方式。如果脚本使用 `cp -al`，本地 session 与 rootfs 必须在同一文件系统；跨文件系统时需要先调整脚本或 rootfs 部署方式。仅增加此参数不能解除硬链接限制。

只领取下载、头环基础质检、视频粗质检和立体矫正均已 `completed`，且 `vio_status=pending` 的数据。不依赖深度估计、手部估计、深度前门禁或千问标注。

每条数据使用自己的 session，输出固定为 `vio/`：

- `report.json`：执行状态、输入身份、有效位姿数量和质量状态。
- `trajectory.npz`：按原视频帧对齐的轨迹。

NPZ 字段为标量 `schema_version`、`quality_status`，逐帧向量 `frame_index`、`right_frame_index`、`timestamp_camera`、`timestamp_imu`、`status`，以及形状为 `(N, 4, 4)` 的 `T_world_imu`、`T_world_camera`。`status=valid` 表示有位姿；无效帧保留原帧号，位姿允许为 NaN。NPZ 总行数不等于有效位姿数量。

宿主机运行时，准备配置和原始 JSONL 仅作为 `vio/.work/run/` 内部中间文件，结束后导出上述两个公开文件。没有位姿的同步帧标记为 `not_estimated`，不猜测具体失败原因。`prepare_only` 仍只生成准备文件，不生成成功交付。

共享盘只回传 `vio/report.json` 和 `vio/trajectory.npz`，不回传内部中间文件。

运行日志为 `annotation/vio_database.log`。时间戳、IMU、标定与视频准备仍使用原算法；起始时间、时长、双目时间匹配容差、IMU 内部/首尾覆盖容差及相机下采样配置仍读取原 YAML。不会自动裁减视频、放宽 IMU 阈值或补充位姿。

领取后写 `processing`。执行成功且两个交付文件为本次新生成，报告中的 session、输入模式和执行状态匹配、有效位姿数大于零、NPZ 帧索引/形状/时间顺序/有效位姿矩阵均正确且数量与报告一致后，才写 `completed`。无效帧不参与位姿矩阵检查。执行异常、空轨迹、输出不完整或校验不一致写 `failed`，不自动写 `block`。`quality_status` 保留原值，文件通过校验不表示轨迹精度合格。

VIO 初始化前可能没有位姿，因此 `completed` 不表示每个视频帧都有位姿，也不代表轨迹精度已经通过质量评估。仅完成输入准备不会标记成功；数据库入口不提供 `prepare_only`。

多终端原子领取，运行期间不持有数据库事务。失败数据需将其 `vio_status` 从 `failed` 改回 `pending` 后再运行；重试沿用原流程重新执行，已有成功文件不能代替本次结果。状态字段已存在，不需要数据库迁移。

## 统一服务器入口（替换旧 runtime-output 包装）

仓库提供 `deploy/vio-run.sh`，保持数据库要求的 `SESSION EMPTY_OUTPUT run` 接口。
调用链为 `database_run → vio-run.sh → chroot → vio.run → OpenVINS`；手动运行从
`vio-run.sh` 开始。新脚本不调用 `/usr/local/bin/vio-runtime` 或 `runtime-output.py`，
项目的 `export_entry.py` 唯一负责交付报告和轨迹。OpenVINS 二进制无需因本次入口调整而重编译。

在服务器上更新宿主仓库后，暂停正在运行的 VIO 任务，再部署同一版本的 VIO 目录和脚本：

```bash
cd /share/project/liuyou/code/roboego-hand-vis
R=/home/humancoindata/vio-rootfs
BACKUP=$(mktemp -d /home/humancoindata/vio-deploy-backup-XXXXXX)
cp -a "$R/opt/roboego-hand-vis/roboego_hand_vis/vio" "$BACKUP/vio"
cp -a /share/project/liuyou/vio-run.sh "$BACKUP/vio-run.sh"
cp -a roboego_hand_vis/vio/. "$R/opt/roboego-hand-vis/roboego_hand_vis/vio/"
install -m 755 roboego_hand_vis/vio/deploy/vio-run.sh /share/project/liuyou/vio-run.sh
printf '备份目录：%s\n' "$BACKUP"
```

rootfs 内仍需现有 Python 依赖和 `/opt/open_vins/ov_msckf/build/roboego_openvins_offline`。
部署前检查同步的 YAML 中 `runtime.system_python` 指向 rootfs 内实际存在的 Python。
脚本默认 rootfs 为 `/home/humancoindata/vio-rootfs`，可通过宿主环境变量 `VIO_ROOTFS` 覆盖。
旧 `vio-runtime doctor/test/prepare` 不属于新入口，直接调用它们仍使用旧脚本逻辑。

单条重跑，无需数据库：

```bash
SESSION=/home/humancoindata/EGOL1/cp_0001_session_20261010_103736
OUT=$(mktemp -d "$SESSION/.vio-manual-XXXXXX")
bash /share/project/liuyou/vio-run.sh "$SESSION" "$OUT" run \
  --input_mode rectified --max_imu_boundary_gap_s 0.3
cat "$OUT/console.log"
```

数据库命令保持原样。数据库任务的算法参数来自**宿主** `--config` 指定的 YAML
（默认宿主 VIO 的 `recommended.yaml`），解析后显式传入 chroot。
手动命令中的参数优先于 rootfs 的 YAML；未传的参数采用 **rootfs** 中的配置。
手动使用 `--config` 时，路径必须能在 rootfs 内访问。
`console.log` 的 `VIO effective config:` 行记录最终生效参数和路径，用它确认容差。

新脚本固定输入、输出和 OpenVINS 安装路径，其余项目 CLI 参数原样传递。
保留原包装的算法环境默认值 `OPENVINS_TIME_OFFSET_MODE=estimate` 和
`OPENVINS_TIMING_BOOTSTRAP=auto`，可用同名宿主环境变量覆盖。
旧包装专用的 `--debug`、`--time-offset-mode`、`--timing-bootstrap` 不再是新入口参数；
调试中间文件默认保留在输出的 `.work/` 中，不再清除。

成功时输出项目原生 `report.json`、`trajectory.npz` 和 `console.log`；失败时保留
`console.log`、`error.log` 及已生成的中间文件，并返回真实退出码。准备阶段失败可能没有
`report.json`，此时直接查看 `error.log`。手动结果不自动写库或替换 session/vio。

脚本增加 rootfs 级 `.vio-run.lock`，新脚本的手动任务和数据库任务会串行运行；旧脚本
不认识此锁，因此切换部署时必须停止旧任务。输入仍用硬链接，要求 session 和 rootfs
位于同一文件系统；输入必须只读。脚本拒绝覆盖已有的 rootfs 输入映射或非空
`data/output`；若异常中断留下这些目录，先检查并备份，再清理后重试。
结果复制失败时保留 rootfs 中的数据；复制成功后清理本次映射和 rootfs 输出。
