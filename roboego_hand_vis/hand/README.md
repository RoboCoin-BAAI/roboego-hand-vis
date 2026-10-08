# Native Hand Launcher

Thin wrapper: MINT inference, followed by EgoHandKit's existing
`scripts/run_without_front.sh mint VIDEO PREDICTION DEPTH OUTPUT`.
No Docker, VIO, depth estimation, Ray or dependency installation is involved.

## Configure Once

Edit `recommended.yaml` or pass `--config /absolute/path/to/local.yaml`.
Set the two source checkouts and absolute Python executable paths (`~` works).
`mint_checkpoint` is relative to the MINT checkout, or absolute. The input video
and depth paths are relative to `--input_dir`; default depth is `depth/`.
For a session using `depth/fast_foundation`, configure that exact directory.
No automatic guessing or upstream data generation is performed.

MINT uses the existing CLI and prediction contract. Record/transfer its local
changes separately: a branch name alone does not capture uncommitted fixes.
EgoHandKit uses feature/mint-frontend plus the native shell configuration patch,
not the Ray branch. The development checkout is `~/EgoHandKit-native-hand`.
Companion patch: EgoHandKit `436c49b` on `feat/native-hand-launcher`.
Once that patch is integrated into your normal feature checkout, configure its
path instead. A worktree does NOT provision untracked model assets or environments.

The outer Python needs PyYAML; algorithms use the configured interpreters.
CUDA/EGL settings are inherited, not repaired. Verify NVIDIA EGL on the native
host separately; Docker's environment fix is not automatically transferred.

## Command

From the native-hand project worktree, after configuration:

```bash
python -m roboego_hand_vis.hand.run \
  --input_dir /path/to/session \
  --output_dir /path/to/new-result \
  --dry-run
```

Remove `--dry-run` to execute. Dry-run only prints commands, cwd and environment
overrides: it does not validate model environments, create outputs, copy an
exporter or run inference. Real execution checks paths/interpreters and exporter
compatibility before inference and requires a NEW output directory.

MINT retains chunked mode, max-frames 10000, target-fps 30, window 32,
window-batch-size 1, no-render. This is not an arbitrary-FPS/long-video adapter;
video/depth alignment must satisfy the existing algorithms.
EgoHandKit still owns all its arguments, including wrist surface compensation
and orientation gating. We do not substitute the older Docker profile.
The external script creates a missing exporter in MINT's scripts directory,
but refuses to overwrite a different existing copy. This explicit side effect
is retained; a full run is not read-only with respect to the MINT checkout.

## Outputs and Limits

All artifacts are retained:

```text
new-result/
  mint_pre/          # MINT predictions and normal metadata
  mint/              # canonical inputs and the existing EgoHandKit result tree
  mint.log
  egohandkit.log
  run_report.json    # commands, paths, start revisions, times, exit status
```

A failed stage stops the chain and leaves logs/partial outputs. A missing or
empty prediction.npz prevents starting EgoHandKit even if MINT exits zero.
`complete` means command completion, not independent pose/pixel/video validation;
quality stays `not_validated`. Ctrl-C/SIGTERM terminates the child process group.
No resume, retry, cleanup or compact-output mode is implemented.

EgoHandKit retains its original test_data cache. A per-checkout advisory lock
serializes this launcher for the same OS user; it does not guard legacy/manual
commands, other users or different checkouts sharing symlinked caches. Do not
run those concurrently. Separate output roots do not imply full cache isolation.

Future output compaction belongs after the two commands and before the final
report. No deletion/copying policy or extra abstraction is added now. Existing
Docker entrypoints are unchanged.

Validation uses CPU tests and a fake interpreter executing the real shell wrapper.
No full native GPU run has been performed. Environments, weights, NVIDIA EGL and
numerical/visual acceptance still need validation on the target host.

Pre-push validation (2026-10-08): 13 native launcher tests, one existing VIO CLI
regression test and one EgoHandKit wrapper test passed. Native tests include
real Python-module/subprocess execution with stand-in algorithms for success
and MINT failure. No model accuracy or performance is inferred from these tests.
The host MINT environment is named `mint-inference`, not `mint`; its CLI help
and PyTorch CUDA availability check passed, and the example config was corrected.
The isolated EgoHandKit checkout lacks HaMeR/MANO weight assets, so full native
GPU inference was not run and no weights were installed or copied.
