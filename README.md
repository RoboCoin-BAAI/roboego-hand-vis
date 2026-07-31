# RoboEgo Hand Vis

Rerun visualization and diagnostics for ego-hand reconstruction from stereo
video. The standard flow is:

1. Resolve a dataset into `dataset_manifest.json`.
2. Check stereo video quality.
3. Estimate depth with FoundationStereo.
4. Run HaMeR ego-hand candidates.
5. Align MANO mesh to depth with a shape-preserved placement.
6. Track, recover short gaps, and smooth the final hand trajectory.
7. Export a clean Rerun file with RGB, depth point cloud, and final meshes.

## Wuxi Pipeline

```bash
bash scripts/run_wuxi_pipeline.sh
rerun tmp/wuxi_hand_vis_000000_000299/wuxi_hand_vis.rrd
```

Common overrides:

```bash
NUM_FRAMES=600 \
RRD_NUM_FRAMES=600 \
bash scripts/run_wuxi_pipeline.sh
```

## Ropedia Pipeline

```bash
bash scripts/run_ropedia_pipeline.sh
rerun tmp/ropedia_hand_vis_000000_000299/ropedia_hand_vis.rrd
```

The final Rerun export keeps the viewer simple:

```text
camera_opencv/camera/left/rgb
camera_opencv/camera/left/depth_vis
camera_opencv/depth/full_scene_points
camera_opencv/hands/ego_left/final
camera_opencv/hands/ego_right/final
camera_opencv/camera/left/rgb/overlay/final/ego_left/mesh_projected_points
camera_opencv/camera/left/rgb/overlay/final/ego_right/mesh_projected_points
```

Generated data, videos, model outputs, and `.rrd` files are intentionally not
tracked by git. Keep them under `tmp/` or external data directories.

## Object Annotation

BAAI yellow-object segmentation, offline semantic status, contact annotations, and Rerun review:

```bash
python scripts/object/run_object_annotation.py \
  --video data/baai/videos/left_rectified.mp4 \
  --depth_dir tmp/baai_foundation_depth_continuous_900 \
  --intrinsics_json tmp/baai_rectified_intrinsics.json \
  --hand_tracks tmp/baai_ego_hand_tracking_900_identity_gate/ego_hand_tracks.json \
  --hand_joints_npz tmp/baai_hand_joints21_identity_bbox_edge_gate/joints_21_camera_opencv.npz \
  --output_dir tmp/baai_object_annotation_900 \
  --backend color_yellow \
  --object_id yellow_object_0 \
  --num_frames 900 \
  --no_qwen

rerun tmp/baai_object_annotation_900/object_annotation.rrd
```

The RGB review layer is `camera_opencv/camera/left/rgb_object_bbox_overlay`.
Rerun output can be large; use `--no_save_rrd` to disable it, or adjust
`--jpeg_quality` (default: `85`) to trade image quality for output size.

## Code Layout

```text
scripts/datasets/resolve_dataset.py
  Dataset-specific loaders -> standard dataset_manifest.json.

scripts/run_foundation_stereo_video.py
  Manifest-driven FoundationStereo depth estimation.

scripts/hand/run_hamer_ego_filter.py
  HaMeR candidates and ego-hand filtering.

scripts/hand/align_mesh_depth.py
  Standard mesh-depth alignment entrypoint.

scripts/hand/track_fill_smooth.py
  Tracking, recovered-depth override, gap fill, and smoothing entrypoint.

scripts/hand/export_final_rerun.py
  Clean final Rerun export.

roboego_hand_vis/egohand/alignment_*.py
  Alignment CLI, IO, geometry, and Rerun helper modules.

roboego_hand_vis/egohand/track_*.py
  Tracking report, debug image, and Rerun helper modules.
```
