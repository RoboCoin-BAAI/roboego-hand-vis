# Double Sphere support

The LeRobot L1/L3 rectification entries accept both pinhole/equidistant and
Kalibr `camera_model: ds`, `distortion_model: none`, `distortion_coeffs: []`.
The stereo pair must use the same model. DS intrinsics are ordered as
`[xi, alpha, fx, fy, cx, cy]`; they must not be interpreted as fisheye coefficients.

DS frames are sampled by projecting virtual rectified pinhole rays back into
the original DS cameras. Rectification uses the actual left-to-right transform.
Projection includes the DS valid-domain check. The output remains a common
zero-distortion pinhole stereo pair with R1/R2/P1/P2/Q and the measured baseline,
so rectified-input VIO and depth consumers retain their existing input format.
The DS equations and projection domain follow the model documented at
https://github.com/VladyslavUsenko/basalt-headers/blob/master/include/basalt/camera/double_sphere_camera.hpp
(implementation is written here in NumPy, without copying the C++ code).

For DS, balance=0 uses a conservative virtual focal length based on central
projection magnification. Raising balance widens the virtual view up to 2x;
the common valid ROI is still cropped. This differs from the fisheye balance
policy. Reuse checks compare the resulting geometry before adopting outputs.

Run the existing L3 command after updating both files:

```bash
python -m roboego_hand_vis.lerobot_stereo_rectification.L3.run \
  --source /data/L3_lerobot/batch_0000014/L3-003-2026-10-16-11-01-53 \
  --workers 1 --ffmpeg /path/to/ffmpeg
```

The changed production files are `L1/compute.py` and
`../stereo_rectification/double_sphere.py` relative to roboego_hand_vis.
This adds DS support to the LeRobot entries; the legacy non-LeRobot
`stereo_rectification/cammodel.py` entry retains its existing model guard.

This does not align timestamps, repair an inaccurate calibration, or enable
raw DS input in OpenVINS. Use the derived rectified pinhole videos for VIO.
Source images, calibration and acquisition timestamps are preserved.
