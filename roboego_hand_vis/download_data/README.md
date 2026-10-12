# L1 数据下载

支持直接输入 KS3 session 路径，或从数据库领取 L1 待下载记录。
两个模式使用相同的下载器，下载器不依赖数据库包。

```text
download_data/
├── downloader.py        # KS3 下载、结果检查、目录发布和校验复用
├── database_adapter.py  # L1 数据库任务领取与状态回写
├── l1.py                # 两种模式的命令行入口
└── ks3/                 # 本地工具与配置，Git 忽略
```

原目录 `downoad_data/` 已统一为 `download_data/`，原 KS3 工具、配置和日志已保留。
默认工具是 `ks3/ks3util-linux-amd64`，配置是 `ks3/ks3util-config`。
其他机器需要准备同样的工具与配置，或通过 `--ks3-tool`、`--ks3-config` 指定。

## 直接下载

在项目根目录执行，传入一个或多个 KS3 session 目录路径：

```bash
python -m roboego_hand_vis.download_data.l1 \
  --output-dir ./data/l1 \
  ks3://baai-eai-datasets/data/cp_0002_session_20260923_112929 \
  ks3://baai-eai-datasets/data/cp_0002_session_20260923_103724 \
  ks3://baai-eai-datasets/data/cp_0002_session_20260923_103758
```

该模式只使用 Python 标准库和 KS3 工具，不读取 `DATABASE_URL`，也不加载数据库依赖。
输入按 session 目录前缀处理，会自动补尾部 `/`，不适用于单文件下载。

## 从数据库下载

先按照[数据库说明](../database/README.md)填写 `database/config/database.yaml`，
并完成建表、迁移和入库。然后显式使用 `--database`：

```bash
python -m roboego_hand_vis.download_data.l1 \
  --database \
  --output-dir ./data/l1 \
  --limit 3
```

数据库模式固定使用 `l1data`，不能同时传入人工 KS3 路径。
`--limit 3` 最多领取三条记录，省略或设为 `0` 时处理当前可领取的全部 pending 记录。
每个进程逐条处理 session；可运行多个进程，共用数据库和相同共享下载根目录。
`--jobs` 默认 `8`，表示单个 session 内同时传输的文件数。

流程：

1. 通过 `DataService.claim("l1data")` 原子领取 pending 记录，状态变为 processing。
2. 使用该记录的 KS3 路径下载；下载期间不持有数据库连接或事务。
3. 下载和发布成功，写入相对路径并更新为 completed。
4. 下载失败，更新为 failed；继续处理其他 pending 记录。

已 completed、processing 或 failed 的记录不会被再次领取。
失败记录需显式调用 `DataService.retry()` 变回 pending 后才能重试。

## 保存路径和完成条件

`--output-dir` 是下载根目录。对于 `ks3://bucket/data/session001`，结果为：

```text
<output-dir>/session001/
```

数据库保存 `local_path = "session001"`。所有使用数据库的服务须约定同一个共享根目录，
使用 `<shared-root>/<local_path>` 访问文件。

文件先下载至根目录下的 `.downloads/<task-id>/payload/`，检查通过后通过同一文件系统内的
目录重命名发布。检查包括工具退出码、失败清单和至少一个下载文件，避免把部分成功
或空前缀误判为完成。成功目录包含 `.ks3_download.json`，记录源路径及本地文件大小。
复用目录时检查记录中的文件存在且大小一致；该校验不替代内容或画面质量检查。

已有目录只有在来源匹配、记录完整且文件大小一致时才复用。
来源不同、缺少完成记录或文件丢失时拒绝覆盖。不同来源同名 session 应使用不同下载根目录。
多个下载进程使用共享根目录内的文件锁协调目录发布；共享存储需要支持文件锁和原子重命名。

下载失败保留临时数据、失败清单和 `.downloads/<task-id>/ks3.log` 供排查，
重试会创建新的临时目录。旧失败目录在确认任务停止后可手动清理。
正常 Ctrl-C 或 SIGTERM 会先停止 KS3 子进程再退出；强制杀死或断电后仍需人工恢复
数据库中的 processing 状态，详见数据库说明。

文件发布与数据库状态回写不是一个原子事务。如果文件已发布但数据库回写失败，
脚本停止；核查并恢复该条记录后，下次下载可校验并复用成功目录。
输出最终 JSON 汇总；下载失败时进程退出码为 `1`。

## 指定工具和配置

```bash
python -m roboego_hand_vis.download_data.l1 \
  --output-dir ./data/l1 \
  --ks3-tool /path/to/ks3util-linux-amd64 \
  --ks3-config /path/to/ks3util-config \
  --jobs 8 \
  ks3://your-bucket/data/session001
```

## 验证

```bash
python -m pytest tests/download_data -q
```

下载测试用本地替代工具验证真实 subprocess 和文件发布，不访问远端 KS3。
数据库适配测试位于 `tests/database/`，需要配置可丢弃数据库的 `TEST_DATABASE_URL`。
