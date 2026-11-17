# 单条简智完整流程测试

结构化后、上传前自动生成 meta/readme.json，使用 build_json/base 共用生成器。生成失败不上传，日志保存在外部状态目录的 build_readme.log。

入口 `scripts/jianzhi/single.py`，在 roboego-hand-vis 仓库根目录执行。只接受已转换为 LeRobot 的 KS3 数据目录（其中有 meta/info.json、meta/annotations.json、data、videos、calibrations），不接受原始 MCAP 或 genrobot dataset token。

## Windows PowerShell

先从下载清单 CSV 的“目标路径”列复制准确的 KS3 session 路径。以下 SOURCE 需要替换为该路径，不要根据文件 ID 猜目录。

```powershell
python scripts/jianzhi/single.py `
  --source "ks3://baai-eai-pwm/jianzhi/实际目录/32位数据ID/" `
  --batch batch_0000000 `
  --local-root D:\data\jianzhi_single_test `
  --state-root D:\data\jianzhi_single_test_state `
  --ks3-tool C:\Users\22563\Desktop\utils\ks3_internal\ks3util.exe `
  --ks3-config C:\Users\22563\Desktop\utils\ks3_internal\ks3util-config-team `
  --argus-repo C:\Users\22563\Desktop\code\argus `
  --ffmpeg D:\Programme\Language\Python\Python3.13.2\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe `
  --dry-run
```

确认计划后去掉 `--dry-run` 执行真实测试。Python 应安装两个仓库所需依赖，包括 numpy、opencv-python、pyarrow、PyYAML、jsonschema 等；KS3 配置须有源目录读取及目标目录写入权限，ffmpeg 须支持 libx264。

文本模型的凭据/服务地址沿用 Argus `review.qwen_endpoint`：通常为 DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL；若设置 XMSXB_API_KEY 则按共享路由规则选择该网关。默认模型 qwen3.7-plus，不自动更换模型。此步骤会实际调用模型并产生费用。

## 实际执行顺序

1. 复用 download_data.base，KS3 内部默认 jobs=64，不额外开下载池。
2. 单进程执行该条前向双目去畸变及立体校正，在原 camera-camchain.yaml 追加推断的头部/body 轴关系，清理旧标定 JSON 和质量报告。
3. 读取 meta/annotations.json，以 qwen3.7-plus 做文本结构化，保留原始细分动作、时间边界；缺失视觉信息留空，通过 HumanCOIN 场景规范和 JSON Schema 校验。
4. 上传整条数据到 `ks3://baai-eai-pwm/jianzhi/wait_for_publish/<batch>/<ID>/`。默认 batch_0000000 用于单条测试，与正式批次区分；同名目标会被上传内容覆盖，请选未使用的批次编号。
5. 上传收据持久化、远端文件大小复核且本地清单未变后，删除 `local-root/<batch>/<ID>/`。local-root 及独立 state-root 不删除。

不需要手动创建 CSV。源数据必须是远端 LeRobot 目录，此脚本没有 MCAP 转换阶段。现有 D:\data\jianzhi 下的样本不会被接管或删除；示例使用独立测试目录。

## 日志和恢复

- 状态：`state-root/<batch>/records/<ID>.json`。
- 日志：`state-root/<batch>/logs/<ID>/download.log`、rectify.log、annotation.log。
- 再次执行同一命令时，从记录的已完成阶段继续；complete 状态直接跳过，不会重新下载。
- error 记录不会自动重试。先检查日志并解决原因，再编辑这条状态 JSON，仅删除 error 字段、保留 status，重新执行。若标注模型请求失败，Argus 的隐藏 `*_text.claim.json` 仍可能记录已发送请求，需先检查其结果再决定是否显式重试，不要直接删除收据导致重复付费。
- 更换源、模型或处理规则做全新测试时，使用新的 state-root 和未使用的 batch 编号。不要用新状态接管旧的已有本地数据目录。
- pipeline.lock 表示正在运行；异常断电后只在确认旧任务已停止时移除它。

脚本已经用模拟测试验证阶段顺序、失败停止、dry-run 无副作用与根目录隔离，未替用户执行真实下载、模型请求、上传或删除。
