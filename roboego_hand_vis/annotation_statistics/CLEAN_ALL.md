# Unified cleaning

Use this entry instead of running four individual cleaning commands:

```bash
cd /share/project/liuyou/humancoin/code/roboego-hand-vis
for dataset in guanglun_depth L3_lerobot; do
  /share/project/liuyou/humancoin/envs/ego/bin/python \
    -m roboego_hand_vis.annotation_statistics.clean_all \
    --input-root "/share/project/liuyou/humancoin/data/$dataset" \
    --mapping "/share/project/liuyou/humancoin/annotation_statistics/task_grouping_coarse_v2/${dataset}_task_mapping.csv" \
    --dataset "$dataset" || break
done
```

One fixed-depth discovery per dataset: batch_*/session/annotation/episode_000000_structured.json. No recursive traversal into videos or data. Each annotation and existing meta/readme.json is read once, transformed in memory and written at most once if changed. Does not create missing readmes, logs or backups. Writes original files directly.

Applies action vocabulary cleaning, object naming rules, scene pair migration and common_task assignment. Preserves task_name and subtask actions. Unmatched tasks receive common_task=null. All transformations in a session are validated before writes begin; errors are printed, other sessions continue, and the command exits nonzero if any session failed. Filesystem write errors can still leave part of a session updated.
