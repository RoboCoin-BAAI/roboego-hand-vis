# L3 world-space hand points

Input: MANUS skeleton and left/right VIVE wrist poses per episode.
Output: observation.hand_left_points.parquet and observation.hand_right_points.parquet,
plus feature declarations in meta/info.json and statistics in meta/episodes_stats.jsonl.
No provenance files or saved source receipts are generated or required.

The transform uses skeleton-root-relative points, maps skeleton X/Y/Z to Tracker Z/X/Y,
and adds t_tracker_wrist = [0, -0.085, -0.058] meters in Tracker coordinates before
applying the Tracker-to-world pose. Both hands use the same alignment and offset.
Output coordinates use the OpenVR TrackingUniverseStanding world frame, in meters.
Tracker RPY is in degrees, with Rz(yaw) @ Ry(pitch) @ Rx(roll).
Position is interpolated linearly and rotation using quaternion SLERP.
The default maximum interpolation bracket is 0.03 seconds; invalid samples are NaN.

Existing complete point files can be reused without provenance. Source hashes and
mounting parameters are not automatically verified. Use --overwrite only when
explicitly intending to recompute from the original skeleton and Tracker inputs.

CPU pipeline: download -> qc -> rectify -> review -> points -> readme.
Pipeline logs and scheduling state stay outside the dataset under state_root.

Explicit migration of existing points (not part of cleanup):

```bash
python scripts/L3/update_points_offset.py --source /path/to/L3_lerobot --batch-range 10:14 --workers 8 --dry-run
```

Removing --dry-run explicitly recomputes existing point files using the default
0.03-second gap. Sessions without point files are skipped. No provenance is read
or generated. This command must not be used when reprocessing is disallowed.
