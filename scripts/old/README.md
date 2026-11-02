# Legacy batch workflows

L1/ and L3/ contain the previous batch orchestrators: run.sh, pipeline.conf,
stage.py, report.py, and the original documentation. L1/ also contains review_retry.sh.

```bash
bash scripts/old/L1/run.sh batch_0000001
bash scripts/old/L3/run.sh batch_0000001
```

VIO setup remains shared at scripts/L1/vio_setup.sh.
Current CPU/GPU workflows remain in scripts/L1/, scripts/L3/, and scripts/pipeline/.
See [current instructions](../pipeline/README.md).
Do not run legacy and current workflows against the same data concurrently.
