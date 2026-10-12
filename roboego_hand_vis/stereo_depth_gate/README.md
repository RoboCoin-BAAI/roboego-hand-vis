# 深度前质量门禁

对矫正后的左右目抽帧检查亮度、清晰度和匹配几何，判断能否用于深度估计。

## 处理指定数据

```bash
python -m roboego_hand_vis.stereo_depth_gate.run --session /path/to/session
```

## 从 L1 数据库批量处理

完成[数据库连接配置](../database/README.md)后，在项目根目录执行：

```bash
python -m roboego_hand_vis.stereo_depth_gate.database_run \
  --shared-root ./data/l1 \
  --limit 3
```

只领取下载、前两项质检和立体矫正均已 `completed`，且 `stereo_depth_gate_status=pending` 的数据。省略 `--limit` 或设为 `0` 时处理当前所有可领取数据；可用 `--config` 指定门禁配置。

领取后写 `processing`，报告 `usable_for_depth=true` 写 `completed`，`false` 写 `block`，执行异常写 `failed`。沿用原门禁的可用性判断：严格检查 `summary_passed=false` 不一定阻止深度估计，应以 `usable_for_depth` 为准。

报告写入每个 session 的 `quality/stereo_depth_gate/stereo_depth_gate_report.json`。门禁不会自动启动深度估计；批次有阻断或失败时退出码为 `1`。

## 阈值配置与报告

两套阈值均在 `recommended.yaml` 中设置；手动入口和数据库入口都支持用 `--config` 指定其他 YAML。

```yaml
thresholds:  # 严格标准
  max_brightness_gap: 35.0
  min_sharpness: 50.0
  max_dark_ratio: 0.25
  max_bright_ratio: 0.25
  min_orb_matches: 30
  max_median_abs_dy: 2.0
  max_p90_abs_dy: 5.0

depth_usable_thresholds:  # 深度放行的几何标准，单位像素
  max_median_abs_dy: 8.0
  max_p90_abs_dy: 70.0
```

深度放行标准的亮度、清晰度、曝光与 ORB 匹配数量仍继承 `thresholds`。未配置宽松几何阈值时，代码默认使用 8 / 70，以兼容原配置。

报告 `stereo_depth_gate_report.json` 中：

- `strict_thresholds` 与 `checks`：完整严格阈值及检查结果。
- `depth_usable_thresholds` 与 `depth_usable_checks`：完整实际放行阈值及检查结果。
- `strict_passed`：是否严格通过；原字段 `summary_passed` 保留。
- `gate_decision`：实际判断依据 `usable_for_depth`、放行结果、对应数据库状态和中文说明。

严格几何不通过、但实际放行检查全部通过时，报告标记 `high_risk`，允许继续，数据库写 `completed`；这不表示深度质量可靠。只要实际放行检查不通过，就阻断并写 `block`。修改配置后需要重新运行质检生成新报告，已有报告和数据库状态不会自动更新。
