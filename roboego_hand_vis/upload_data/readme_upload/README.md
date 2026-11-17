# HumanCOIN 数据集发布脚本

已集成到 RoboEgo Hand Vis 的 `upload_data.readme_upload` 模块。从仓库根目录使用下述命令运行，也可进入本目录直接运行脚本。仅负责本地检查、README 生成、上传和远端删除，不包含网页、定时任务或跨仓库授权同步。

## 文件

| 文件 | 作用 |
|---|---|
| `check_load.py` | 使用 HumanCOIN 读取首、中、末帧，检查加载情况 |
| `gen_readme.py` | 根据元数据生成 README，不自动加载数据 |
| `upload2hub.py` | 上传 HF / 魔搭，支持新建、批量与仅更新 README |
| `delete_from_hub.py` | 删除指定远端 dataset 仓库及历史 |
| `_common.py` | 共用的命名、文件筛选、凭据与错误处理 |
| `config/project.json` | 作者、联系邮箱、引用及数据许可证 |
| `config/hf.yaml`、`config/ms.yaml` | 可选上传配置示例 |
| `tests/` | 模拟客户端测试，不操作远端 |

## 安装

Python 3.10+：

```bash
pip install -r roboego_hand_vis/upload_data/readme_upload/requirements.txt
# 仅需要 check_load.py 时，额外安装：
pip install -r roboego_hand_vis/upload_data/readme_upload/requirements-check.txt
```

数据读取工具：[FlagOpen/HumanCOIN](https://github.com/FlagOpen/HumanCOIN)。数据门户：[HumanCOIN-DATA](https://flagopen.github.io/HumanCOIN-DataManager/)。

## 1. 加载检查（可选）

```bash
python -m roboego_hand_vis.upload_data.readme_upload.check_load --dataset /path/to/dataset
```

检查主时间轴帧数、样本时间戳和 observation 特征，通过 HumanCOIN/PyAV 读取首、中、末帧。异常返回非零退出码；不修改数据、不写 README。这是抽样检查，不是逐帧验证。

## 2. 生成 README

```bash
python -m roboego_hand_vis.upload_data.readme_upload.gen_readme --dataset /path/to/dataset
# 覆盖已有文件：
python -m roboego_hand_vis.upload_data.readme_upload.gen_readme --dataset /path/to/dataset --force
```

读取 `meta/readme.json`、`meta/info.json`、可选的 `meta/tasks.jsonl` 和 `config/project.json`，默认生成 `<dataset>/README.md`。包含任务、场景、传感器、时间对齐、加载示例、作者及引用；文件数和体积从本地重算。`--output` 可指定输出路径，`--project-config` 可替换项目配置。

数据集许可证为 **CC BY-NC 4.0**（`license: cc-by-nc-4.0`），不代表工具代码的许可证。README 带 HF 的 `Name`、`Email`、`Intended use` 文本申请表单；HF 账号信息收集由平台设置控制。

需要“检查通过才生成”时自行串联：

```bash
python -m roboego_hand_vis.upload_data.readme_upload.check_load --dataset /path/to/dataset && \
python -m roboego_hand_vis.upload_data.readme_upload.gen_readme --dataset /path/to/dataset --force
```

## 3. 上传

凭据使用 `HF_TOKEN` / `MODELSCOPE_TOKEN` 环境变量或平台已有登录缓存；魔搭也支持 `MS_TOKEN`。不要把 Token 写进 YAML 或版本库。需要时可在 Bash 中隐藏输入：

```bash
read -rsp 'HF Token: ' HF_TOKEN; echo; export HF_TOKEN
read -rsp 'ModelScope Token: ' MODELSCOPE_TOKEN; echo; export MODELSCOPE_TOKEN
```

先预览，再分别上传两个平台：

```bash
python -m roboego_hand_vis.upload_data.readme_upload.upload2hub --hub hf --folders /path/to/dataset --create-repo --dry-run
python -m roboego_hand_vis.upload_data.readme_upload.upload2hub --hub hf --folders /path/to/dataset --create-repo
python -m roboego_hand_vis.upload_data.readme_upload.upload2hub --hub ms --folders /path/to/dataset --create-repo
```

默认目标为 `HumanCOIN/<dataset_name>`，名称读取 `meta/readme.json`，缺少该文件时使用目录名。`--create-repo` 允许创建不存在的仓库；已有仓库只更新同名文件，不删除远端独有文件。上传不自动生成 README，也不自动运行加载检查。

常用选项：

| 参数 | 用途 |
|---|---|
| `--folders A B` | 一次上传多个本地数据集，每个对应一个仓库 |
| `--repo-id owner/name` | 显式指定目标，仅限一个本地目录 |
| `--readme-only` | 仅上传根目录 README |
| `--allow` / `--ignore` | 相对路径 glob 筛选；生成 README 也支持相同参数 |
| `--config roboego_hand_vis/upload_data/readme_upload/config/ms.yaml` | 从 YAML 读取参数，命令行优先 |
| `--max-workers 8` | 魔搭并发；HF 使用 SDK 默认并发 |
| `--max-retries 3` | 总上传尝试次数 |
| `--skip-errors` | 批次中失败后继续，最终仍返回非零状态 |
| `--dry-run` | 只显示计划，不联网、不修改远端 |

默认排除 `.git`、缓存、`backup`、`.env` 等，不上传软链接。新 HF 仓库可用 `--create-repo --private` 保持私有。

**权限与平台适配：** HF 上传前设置 Auto。魔搭创建、上传时设置 Gated Auto，上传前后回读验证，未生效则报错。HF 上传完整 README；魔搭用临时副本去掉 `gated` 和 `extra_gated_*`，保留许可证、通用元数据与正文。本地 README 不变，大文件优先硬链接暂存，临时目录自动清理。HF 表单不代表魔搭支持相同字段。上传后验证失败时文件可能已在远端，不会自行删除。

## 4. 删除远端仓库

```bash
python -m roboego_hand_vis.upload_data.readme_upload.delete_from_hub --hub both --repo-id HumanCOIN/DATASET_NAME --dry-run
python -m roboego_hand_vis.upload_data.readme_upload.delete_from_hub --hub both --repo-id HumanCOIN/DATASET_NAME
```

`--hub` 支持 `hf`、`ms`、`both`，`--repo-id` 可写多个完整仓库名称。缺少环境变量时隐藏询问各平台 Token；删除需要相应管理权限。不再额外确认，非预览命令会删除远端仓库及历史，保留本地数据。删除后回读验证，不自动重试删除请求；跨平台不是事务，某个平台失败时另一个仍会尝试。

重传流程：删除成功 → 生成 README → `upload2hub.py --create-repo`。普通更新不需要先删除。

上传依赖单独安装，不会自动运行流水线。Hugging Face SDK 版本范围允许 0.x / 1.x，避免与主项目的 1.x 版本约束冲突。接口签名测试已在本机 0.35.3 上通过；1.x 和魔搭 SDK 尚未在本机验证。默认项目配置随 Python 包一起安装。

## 测试

```bash
python -m unittest discover -s tests/readme_upload -v
```

本地脚本不会读取 GitHub Actions Secrets，不存储 Token 或用户申请信息。用 `--help` 查看各入口完整参数。
