# Headband Session Quality Report Design

## Goal

Check a raw RoboEgo headband session and write an initial quality report inside the source session:

```text
data/session_20260804_224442/annotation/quality_report.json
```

This step does not convert data, does not create LeRobot layout folders, does not copy videos, and does not decode every video frame.

## Runtime

Use the `py312` conda environment:

```bash
conda run -n py312 python scripts/export/check_headband_data.py \
  --session data/session_20260804_224442 \
  --overwrite
```

## Inputs

The checker expects:

```text
meta.json
videos/left.mp4
videos/right.mp4
videos/bleft.mp4
videos/bright.mp4
timestamps/left_timestamps.txt
timestamps/right_timestamps.txt
timestamps/bleft_timestamps.txt
timestamps/bright_timestamps.txt
imu/imu0.csv
```

Audio and calibration files are not processed by this first quality report.

## Report Contents

`quality_report.json` records:

- source session path
- selected camera streams
- video container metadata
- timestamp file paths
- selected frame window
- fps and image size
- per-camera timestamp duplicate count, duplicate rate, and first duplicate index
- per-camera timestamp alignment statistics against `left`
- IMU timestamp sample count, first/last timestamp, duration, estimated fps, duplicate count/rate, first duplicate index
- IMU start-time delta to each selected camera stream, computed as `imu_first_timestamp - camera_first_timestamp`
- default timestamp tolerance, equal to one frame interval plus 1 ms when not provided

Alignment mismatches are recorded, not rejected. The checker only fails for source integrity problems such as missing files, malformed timestamps, decreasing timestamps, unreadable videos, invalid video dimensions, incompatible frame counts, invalid frame window, or output report already existing without `--overwrite`.

## CLI Options

```text
--session PATH
--output_root PATH
--start N
--num_frames N
--timestamp_tolerance_s SECONDS
--camera_stream left --camera_stream right ...
--overwrite
```

When `--output_root` is omitted, the report root is the source session directory.

## Testing

Focused tests cover timestamp parsing, duplicate-rate summaries, alignment summaries, IMU timestamp summaries, source validation, default report location, CLI defaults, and report writing without creating `meta/`, `data/`, or LeRobot-style `videos/` folders.
