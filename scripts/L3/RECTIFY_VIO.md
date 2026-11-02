# Explicit-list L3 rectification and stereo VIO

Run from the roboego-hand-vis repository after syncing the DS rectifier and
model-aware QC changes. This script does not rerun QC or modify pipeline state.
Only use a list whose input prerequisites have been checked.

```bash
/share/project/liuyou/humancoin/envs/ego/bin/python scripts/L3/rectify_vio.py \
  --sessions-file /share/project/liuyou/humancoin/data/batch14_intrinsics_failed.sessions.json \
  --openvins-root /share/project/liuyou/humancoin/code/open_vins \
  --ffmpeg /share/project/liuyou/humancoin/envs/ego/bin/ffmpeg \
  --workers 1
```

For every selected session, rectification runs first. Only a zero exit code
allows VIO to start. A failed session does not stop other sessions. VIO quality
failures retain their original diagnostics and count as failures in the summary.
Rectification validates and reuses compatible outputs; VIO runs again, updating
the corresponding episode outputs using the existing publisher. --force-rectify
forces rectification recomputation.

Default policy is normal VIO (no ZUPT). Use --zupt for every run, or
--zupt-fallback for normal VIO followed by a ZUPT retry after backend/quality
failure. The existing output directories remain vio/ and vio_zupt/.

--workers controls concurrent sessions; stages within each session are serial.
Increase cautiously: each backend can use multiple CPU threads.

A unique run directory is created under the session-list parent's
rectify_vio_runs/ (override the parent with --log-dir). It contains sessions.json,
per-session rectify.log/vio.log/result.json, results.jsonl and summary.json.
Child output streams to the console as it is emitted. No batch was run locally;
tests exercise stage ordering and error handling with mocked stage execution.
