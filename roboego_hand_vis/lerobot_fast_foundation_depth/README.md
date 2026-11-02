# LeRobot Fast-FoundationStereo 深度估计

- [L1](L1/README.md)：适配 L1 LeRobot 的矫正视频、标定 manifest 和时间戳。
- [L3](L3/README.md)：适配 L3 头环左右矫正视频和 episode manifest。

原 `fast_foundation_depth` 保持不变，新目录复用其模型推理 API。


Before rectified stereo inference, L1/L3 share a first-frame quality gate.
SIFT ratio-test matches must be mutual, with >=30 matches spread over >=4 cells
of a 4x3 grid. Median absolute vertical error must be <=1.5 pixels and P95 <=3
pixels at the rectified image resolution. These are initial heuristic thresholds,
not calibrated guarantees. No vertical-error filtering is applied before measuring.
Reports are cached at annotation/stereo_quality/<chunk>_<episode>.json and reused
when video paths, sizes, mtimes and policy match. Warning/inconclusive results
block depth/VIO with a nonzero exit, leaving rectification and semantic annotation
unaffected. Only frame 0 is assessed; later frame quality is not certified.
