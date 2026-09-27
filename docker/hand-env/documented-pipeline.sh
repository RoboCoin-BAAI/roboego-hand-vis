#!/usr/bin/env bash
# Run inside the existing image via run.sh; algorithm source stays mounted.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo 'Usage: documented-pipeline.sh VIDEO DEPTH_DIR NEW_OUTPUT_ROOT [MAX_FRAMES=10000]' >&2
    exit 2
fi
video=$1
depth=$2
output=$3
max_frames=${4:-10000}
mint=/home/user/wuji-ego-mint
ego=/home/user/EgoHandKit
mint_python=/home/user/envs/mint/bin/python
ego_python=/home/user/envs/egohandkit/bin/python
[[ -f "$video" && -d "$depth" ]] || { echo 'Video/depth input missing' >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "Use a new output directory: $output" >&2; exit 1; }
# The original wrapper copies this exporter into MINT. Read-only mounts require
# the two copies to agree instead of mutating the source during inference.
cmp "$ego/scripts/export_egohandkit_cache_mint.py" "$mint/scripts/export_egohandkit_cache.py"
"$mint_python" -c '
import cv2, sys
cap = cv2.VideoCapture(sys.argv[1])
count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
cap.release()
assert 0 < count <= int(sys.argv[2]), "Trim RGB and matching depth first; exporter needs every video frame"
assert abs(fps - 30) < 0.001, "Documented profile requires 30 FPS; do not silently resample depth-aligned video"
print(f"Input: {count} frames at {fps} FPS")
' "$video" "$max_frames"
mkdir -p "$output/mint/canonical_inputs"
exec > >(tee "$output/pipeline.log") 2>&1
trap 'status=$?; echo "Exit status: $status"' EXIT
observations="$output/mint/canonical_inputs/$(basename "${video%.*}")_mint.observations.pkl"
set -x
cd "$mint"
"$mint_python" -m mint infer --input "$video" \
    --checkpoint /home/user/model/mint/model.safetensors --output "$output/mint_pre" \
    --camera-mode chunked --max-frames "$max_frames" --target-fps 30 \
    --window 32 --window-batch-size 1 --no-render
"$ego_python" "$mint/scripts/export_egohandkit_cache.py" \
    --source "$video" --raw "$output/mint_pre/prediction.npz" --out "$observations"
cd "$ego"
"$ego_python" run.py \
    --input "$video" --frontend canonical --observations "$observations" \
    --backend hamer --depth_gate --depth_dir "$depth" --depth_max_m 1.0 \
    --mint_depth_sensor_anchor --motion_gate --mint_3d_consistency_gate \
    --hmr_primary_policy --hmr_duplicate_image_gate --hmr_partial_hand_recovery \
    --hmr_stable_wrist_anchor --endpoint_wrist_gate --endpoint_wrist_max_deg 100 \
    --final_joints_smoother --hand_tracking_parquet --render_hand_tracking_parquet \
    --force_detect --gpu 0 --output_root "$output/mint"
