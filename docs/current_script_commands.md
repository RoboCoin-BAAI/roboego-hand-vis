# Ropedia 当前保留流程

本文档只记录当前确认保留的 300 帧 Ropedia 手部 mesh + depth 可视化流程。旧的 7/29 帧调试命令、不同旋转/平滑/anchor 实验版本已不再作为主线使用。

## 最终 RRD

```bash
rerun tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_temporal_gated_final/ropedia_hamer_new_method_shape_preserved_area0007_temporal_gated_final.rrd
```

## 当前保留的 tmp 数据

```text
tmp/ropedia_foundation_depth_continuous_000000_000299
  FoundationStereo 0-299 帧 depth 输出。

tmp/ropedia_hamer_ego_filter_000000_000299_tuned
  HaMeR 0-299 帧 mesh、HaMeR overlay、原始候选结果。

tmp/ropedia_hamer_ego_filter_000000_000299_recovered_area0007_temporal_gated
  放宽 size 后的 recovered candidates，并移除 temporal jump 特别大的 recovered 候选。

tmp/ropedia_ego_hand_tracking_000000_000299_area0007_temporal_gated_fixed
  当前保留的 tracks json/csv。已删除该目录里的调试 RRD 和 debug frames。

tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_temporal_gated_final
  当前最终 RRD 和 hand-depth-mesh alignment report。
```

## 最终可视化生成命令

推荐直接运行这个固定脚本，避免漏掉 mask/depth-lift 参数：

```bash
source /home/liuyou/miniconda3/etc/profile.d/conda.sh
conda activate ego
bash scripts/run_ropedia_shape_preserved_final_rrd.sh
```

脚本内部等价于下面这条完整命令：

```bash
python scripts/view_ropedia_hamer_depth_alignment_rerun.py \
  --candidates_json tmp/ropedia_hamer_ego_filter_000000_000299_recovered_area0007_temporal_gated/ego_hand_candidates_recovered_temporal_gated.json \
  --tracks_json tmp/ropedia_ego_hand_tracking_000000_000299_area0007_temporal_gated_fixed/ego_hand_tracks.json \
  --tracks_current_best_mode fill_missing \
  --depth_dir tmp/ropedia_foundation_depth_continuous_000000_000299 \
  --video /home/liuyou/Documents/ropedia_data/stereo_left.mp4 \
  --hamer_overlay_dir tmp/ropedia_hamer_ego_filter_000000_000299_tuned/hamer_overlay \
  --output_dir tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_temporal_gated_final \
  --save_rrd tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_temporal_gated_final/ropedia_hamer_new_method_shape_preserved_area0007_temporal_gated_final.rrd \
  --scene_point_stride 4 \
  --max_scene_points 90000 \
  --display_mesh_rotate_axis none \
  --display_mesh_rotate_degrees 0 \
  --display_mesh_rotate_about camera \
  --display_mesh_translate 0 0 0 \
  --new_method_only \
  --mesh_mask_source hamer_overlay_diff \
  --mesh_mask_padding_px 2 \
  --overlay_diff_threshold 8 \
  --overlay_diff_search_margin_px 24 \
  --mask_depth_band_half_width_m 0.12 \
  --depth_lift_window_radius_px 2 \
  --depth_lift_max_delta_m 0.12
```

## 当前 mesh-depth 逻辑

当前保留的 mesh 是 `shape_preserved_depth_aligned_mano` / tracking 后的 `current_best_mesh`：

```text
HaMeR mesh 原始 MANO 手形状
HaMeR mesh 投影到 RGB 得到 2D projected pixels
depth mask 内取 hand depth points
使用 hand depth points 的 median z
使用 projected pixels 的 median u/v
通过 Ropedia left camera K 反投影得到 target anchor
将 HaMeR/MANO mesh 整体 3D translation 到 target anchor
```

这条路线保留 HaMeR/MANO 的手部姿态和形状，用 FoundationStereo depth 提供 3D 位置。

## 必要参数含义

```text
--tracks_current_best_mode fill_missing
  正常检测帧继续使用当前帧的 shape_preserved_depth_aligned_mano。
  只有缺失帧才从 tracking/smoothing 的结果补 current_best_mesh。

--mesh_mask_source hamer_overlay_diff
  使用 HaMeR 官方 overlay 与 RGB 原图的差异得到真实渲染手部区域。
  这是当前 final 版本最关键的手部 depth mask 来源。

--mesh_mask_padding_px 2
  对 overlay diff mask 做很小的外扩，减少漏掉边缘手指。

--overlay_diff_threshold 8
  RGB 与 HaMeR overlay 的差异超过该阈值时认为是 mesh 渲染区域。

--overlay_diff_search_margin_px 24
  只在每只手 bbox 附近搜索 overlay diff，避免另一只手或背景干扰。

--mask_depth_band_half_width_m 0.12
  在 mask 内按深度中值保留 +/- 0.12m 的点，过滤袖子、物体和深度离群点。

--depth_lift_window_radius_px 2
  将 mesh 投影点提升到深度时，在投影像素附近 2px 窗口取稳健深度。

--depth_lift_max_delta_m 0.12
  mesh 顶点深度与 hand mask 中值相差超过 0.12m 的点不信任。

--display_mesh_rotate_axis none / --display_mesh_rotate_degrees 0
  当前 final 版本没有额外旋转 display mesh。后续候选必须保持这个参数，否则 Rerun 中手部会出现方向颠倒。

--display_mesh_translate 0 0 0
  当前 final 版本没有额外平移 display mesh。后续候选必须保持这个参数，否则视觉上会和 final 不一致。
```

## 当前 300 帧统计

来自当前 final report：

```text
ego_left:
  normal: 266
  recovered: 27
  missing: 7, frames 176-182

ego_right:
  normal: 260
  recovered: 40
  missing: 0

current_best_mesh_source:
  shape_preserved_depth_aligned_mano: 569
  tracking_smoothed_shape_preserved_mano: 24

shape_preserved mesh->depth median:
  ego_left normal: 0.0161m
  ego_left recovered: 0.0162m
  ego_right normal: 0.0167m
  ego_right recovered: 0.0189m
```

## recovered 深度后续处理原则

当前 final 版本先作为基准，不再整体替换 normal 手的 mesh/depth 对齐。

当前 recovered z 插值候选脚本：

```bash
source /home/liuyou/miniconda3/etc/profile.d/conda.sh
conda activate ego
bash scripts/run_ropedia_shape_preserved_recovered_z_interp_rrd.sh
```

分层可视化候选脚本：

```bash
source /home/liuyou/miniconda3/etc/profile.d/conda.sh
conda activate ego
bash scripts/run_ropedia_shape_preserved_recovered_z_interp_layered_rrd.sh
```

候选 RRD：

```bash
rerun tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_recovered_z_interp_candidate/ropedia_hamer_new_method_shape_preserved_area0007_recovered_z_interp.rrd
```

分层候选 RRD：

```bash
rerun tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_recovered_z_interp_layered/ropedia_hamer_new_method_shape_preserved_area0007_recovered_z_interp_layered.rrd
```

插值明细：

```text
tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_recovered_z_interp_candidate/recovered_depth_overrides.csv
tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_recovered_z_interp_candidate/recovered_depth_overrides.json
```

这个候选只针对 `is_recovered=true` 的帧做修正：

```text
normal 手:
  继续使用当前帧 overlay diff mask + depth median z 的 shape_preserved 对齐。

recovered 手:
  保留 HaMeR/MANO 姿态和 RGB 上的 x/y 位置。
  不直接信任 recovered 当前帧的小 mask depth anchor。
  优先从同侧相邻 normal 帧插值得到 z。
  插值只用于深度 z，不改变手形状和 2D 位置。
  如果插值后与邻近帧跳变过大，标记为低可信，而不是覆盖原 final。
```

当前候选统计：

```text
recovered z overrides: 67

normal 帧:
  与 final 版本保持一致。

current_best_mesh_source:
  shape_preserved_depth_aligned_mano: 569
  tracking_smoothed_shape_preserved_mano: 24

注意:
  recovered z 插值版不一定让 mesh-depth 数值误差更小。
  它的目标是让 recovered 手的深度不要在相邻帧之间乱跳。
  是否替换 final 需要以 Rerun 中 recovered 帧的视觉连续性为准。
```

## Rerun 分层显示

分层候选 RRD 会保留旧的 `current_best_mesh` 路径，同时新增标准分层：

```text
camera_display/hands/ego_left/layers/normal_mesh
camera_display/hands/ego_left/layers/recovered_mesh
camera_display/hands/ego_left/layers/smooth_mesh
camera_display/hands/ego_left/layers/current_mesh

camera_display/hands/ego_right/layers/normal_mesh
camera_display/hands/ego_right/layers/recovered_mesh
camera_display/hands/ego_right/layers/smooth_mesh
camera_display/hands/ego_right/layers/current_mesh
```

RGB 上也有对应分层：

```text
camera_display/camera/left/rgb/layers/normal_mesh_overlay
camera_display/camera/left/rgb/layers/recovered_mesh_overlay
camera_display/camera/left/rgb/layers/smooth_mesh_overlay
camera_display/camera/left/rgb/layers/current_mesh_overlay
```

颜色语义：

```text
normal: orange/yellow, 当前帧真实检测 + depth 对齐。
recovered: magenta/pink, 放宽规则恢复的候选 + normal 深度插值 z。
smooth: blue, 当前帧无可靠候选时由 tracking/interpolation 补帧。
current: green, 最终采用结果，优先级 normal > recovered > smooth。
```

## 当前不保留的实验

不再保留以下实验输出：

```text
7/29 帧小样本调试 RRD
不同 y90/y180/zdown 旋转实验
mask anchor / finger anchor / z-only 单独实验 RRD
recovered depth temporal repair 实验版本
relaxed_size 单独版本
fill_missing_old_alignment / smooth_old_format / old_exact_plus_fill_missing 版本
```
