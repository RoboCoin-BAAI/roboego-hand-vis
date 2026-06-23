# RoboEgo Hand Vis

Rerun visualization and diagnostics for Ropedia ego-hand reconstruction.

This repository is a slimmed hand-visualization pipeline extracted from
`RoboEgoPipe`. It keeps only the code needed to inspect HaMeR/MANO hand meshes,
FoundationStereo depth, recovered hand candidates, temporal tracking, and
layered Rerun output.

## Main Output

Open the layered 300-frame Rerun file after generating local `tmp/` outputs:

```bash
rerun tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_recovered_z_interp_layered/ropedia_hamer_new_method_shape_preserved_area0007_recovered_z_interp_layered.rrd
```

## Main Pipeline

```bash
bash scripts/run_ropedia_shape_preserved_recovered_z_interp_layered_rrd.sh
```

The layered Rerun output contains:

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

See `docs/current_script_commands.md` for the full command history and
parameter notes.

Generated data, videos, model outputs, and `.rrd` files are intentionally not
tracked by git. Keep them under `tmp/` or external data directories.
