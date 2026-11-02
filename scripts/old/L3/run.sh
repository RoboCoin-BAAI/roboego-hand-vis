#!/usr/bin/env bash
# Linux / Bash >= 4. Commands and defaults follow the six L3 entry points.
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${L3_CONFIG:-$HERE/pipeline.conf}
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
source "$CONFIG"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export PYTHONPATH="$CODE_ROOT/argus${PYTHONPATH:+:$PYTHONPATH}"
VIS="$CODE_ROOT/roboego-hand-vis/roboego_hand_vis"
STAGES=(download qc rectify depth vio review)
for tool in sha256sum setsid flock; do command -v "$tool" >/dev/null; done
mkdir -p "$STATE_ROOT" "$OUTPUT_ROOT"
# Lock the output root, including when different state directories are used.
exec 9>"$OUTPUT_ROOT/.l3_pipeline.lock"
flock -n 9 || { echo 'Another pipeline owns this output root' >&2; exit 2; }
[[ -x "$PYTHON" && -x "$FFMPEG" && -x "$FFPROBE" && -x "$KS3_TOOL" && -f "$KS3_CONFIG" ]]
[[ -n ${DASHSCOPE_API_KEY:-} ]] || { echo 'Export DASHSCOPE_API_KEY first' >&2; exit 2; }
for n in "$QC_WORKERS" "$RECTIFY_WORKERS" "$VIO_WORKERS" "$REVIEW_WORKERS" "$MAX_INFLIGHT"; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo 'Worker/window values must be positive integers' >&2; exit 2; }
done
[[ "$RETRIES" =~ ^[0-9]+$ ]]
shopt -s nullglob
CSVS=("$INPUT_ROOT"/batch_*.csv)
if (( $# )); then
    CSVS=()
    for batch in "$@"; do
        [[ "$batch" =~ ^batch_[0-9]+$ && -f "$INPUT_ROOT/$batch.csv" ]] || { echo "Invalid batch: $batch" >&2; exit 2; }
        CSVS+=("$INPUT_ROOT/$batch.csv")
    done
fi
(( ${#CSVS[@]} )) || { echo 'No batch CSVs found' >&2; exit 2; }
if [[ ${VIO_SETUP:-1} == 1 ]]; then
    "$PYTHON" -c 'import cv2, numpy, pyarrow, yaml; print("VIO Python dependencies available")'
    setup_log="$STATE_ROOT/vio_setup.$(date +%s%N).log"
    echo "Preparing VIO dependencies/build; log=$setup_log"
    OPENVINS_ROOT="$OPENVINS_ROOT" VIO_INSTALL_DEPS="${VIO_INSTALL_DEPS:-1}" \
      VIO_BUILD_JOBS="${VIO_BUILD_JOBS:-2}" \
      bash "$HERE/../../L1/vio_setup.sh" 2>&1 | tee "$setup_log"
fi
BASE_HASH=$(cat "$CONFIG" "$0" "$HERE/stage.py" | sha256sum | cut -d' ' -f1)
DIRS=()
for csv in "${CSVS[@]}"; do
    batch=$(basename "$csv" .csv)
    hash=$( { printf '%s\n' "$BASE_HASH"; sha256sum "$csv"; } | sha256sum | cut -d' ' -f1)
    dir="$STATE_ROOT/$batch/$hash"
    mkdir -p "$dir"
    # Failed stages are retried on the next invocation; successful ones remain.
    rm -f -- "$dir"/*.failed "$dir"/*.blocked
    DIRS+=("$dir")
done
command_for() {
    local stage=$1 batch=$2 src="$OUTPUT_ROOT/$2"
    case "$stage" in
    download) CMD=("$PYTHON" "$VIS/download_data/L3/run.py" --batch-csv "$INPUT_ROOT/$batch.csv" --ks3-tool "$KS3_TOOL" --ks3-config "$KS3_CONFIG" --output-dir "$src");;
    qc) CMD=("$PYTHON" "$VIS/lerobot_headband_integrity/L3/run.py" --source "$src" --workers "$QC_WORKERS" --ffprobe "$FFPROBE" --calibration-root "$OUTPUT_ROOT");;
    rectify) CMD=("$PYTHON" "$VIS/lerobot_stereo_rectification/L3/run.py" --source "$src" --workers "$RECTIFY_WORKERS" --ffmpeg "$FFMPEG");;
    depth) CMD=("$PYTHON" "$VIS/lerobot_fast_foundation_depth/L3/run.py" --source "$src" --model-root "$MODEL_ROOT" --ckpt "$DEPTH_CKPT" --engine throughput --gpus "$DEPTH_GPUS" --num-frames auto --depth-format hevc12 --ffmpeg "$FFMPEG" --hevc12-preset ultrafast --cpu-threads 2 --encode-threads 2 --no-save-vis);;
    vio) CMD=("$PYTHON" "$VIS/lerobot_vio/L3/run.py" --source "$src" --openvins-root "$OPENVINS_ROOT" --workers "$VIO_WORKERS" --input-mode rectified --stereo-tolerance-s 0.001 --time-offset-mode estimate --timing-bootstrap auto "$VIO_POLICY");;
    review) CMD=("$PYTHON" "$CODE_ROOT/argus/review/L3/run.py" --source "$src" --workers "$REVIEW_WORKERS" --jobs 1 --no-board --model qwen3.8-27b --reasoning none --sample-fps 2 --max-tokens 16384 --concurrency 1 --cap 0 --estimated-cap-cny "$REVIEW_CAP_CNY");;
    esac
    if [[ "$stage" != download ]]; then
        local input
        case "$stage" in
        qc) input="$INPUT_ROOT/$batch.csv";;
        rectify) input="$dir/qc.sessions.json";;
        *) input="$dir/rectify.sessions.json";;
        esac
        CMD=("$PYTHON" "$HERE/stage.py" --stage "$stage" --state "$dir"
             --source "$src" --input "$input" --raw-source "$INPUT_ROOT/$batch" -- "${CMD[@]}")
    fi
}
terminal() { [[ -f "$1.done" || -f "$1.failed" || -f "$1.blocked" ]]; }
lane() (
    stage=$1; child=''
    trap 'if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; exit 130' INT TERM
    for i in "${!CSVS[@]}"; do
        dir=${DIRS[$i]}; batch=$(basename "${CSVS[$i]}" .csv)
        [[ -f "$dir/$stage.done" ]] && continue
        dep=''
        case "$stage" in qc) dep=download;; rectify) dep=qc;; depth|vio|review) dep=rectify;; esac
        if [[ -n "$dep" ]]; then
            until terminal "$dir/$dep"; do sleep 2; done
            if [[ ! -f "$dir/$dep.done" ]]; then touch "$dir/$stage.blocked"; continue; fi
        fi
        # Backpressure: do not download unlimited raw + derived batches.
        if [[ "$stage" == download ]] && (( i >= MAX_INFLIGHT )); then
            older=${DIRS[$((i-MAX_INFLIGHT))]}
            for tail in depth vio review; do until terminal "$older/$tail"; do sleep 2; done; done
        fi
        command_for "$stage" "$batch"
        printf -v command '%q ' "${CMD[@]}"
        printf '%s\n' "$command" > "$dir/$stage.command.sh"
        attempts=$((RETRIES+1))
        # Paid/API jobs are only retried by an explicit rerun, using Argus receipts.
        [[ "$stage" == review ]] && attempts=1
        ok=0
        for ((attempt=1; attempt<=attempts; attempt++)); do
            echo "[$(date -Is)] $batch $stage attempt=$attempt"
            log="$dir/$stage.$(date +%s%N).log"
            started=$SECONDS
            setsid "${CMD[@]}" >"$log" 2>&1 & child=$!
            rc=0; wait "$child" || rc=$?; child=''
            printf '%s exit=%s duration_s=%s log=%s\n' "$(date -Is)" "$rc" "$((SECONDS-started))" "$log" >> "$dir/$stage.history"
            if ((rc == 0)); then
                touch "$dir/$stage.done"; ok=1
                echo "[$(date -Is)] DONE $batch $stage"
                break
            fi
            echo "FAILED $batch $stage exit=$rc log=$log" >&2
            if ((attempt<attempts)); then sleep 10; fi
        done
        if (( ! ok )); then touch "$dir/$stage.failed"; fi
    done
)
PIDS=()
cleanup() {
    trap '' INT TERM
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap 'cleanup; exit 130' INT TERM
echo "Batches=${#CSVS[@]}, window=$MAX_INFLIGHT; state=$STATE_ROOT"
for stage in "${STAGES[@]}"; do lane "$stage" & PIDS+=("$!"); done
# Unexpected lane failure must stop dependent lanes instead of hanging forever.
while :; do
    running=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then running=1
        elif wait "$pid"; then :
        else echo "Pipeline lane crashed: $pid" >&2; cleanup; exit 1; fi
    done
    ((running)) || break
    sleep 2
done
failed=0
for i in "${!DIRS[@]}"; do
    batch=$(basename "${CSVS[$i]}" .csv)
    for stage in "${STAGES[@]}"; do
        status=done
        if [[ -f "${DIRS[$i]}/$stage.partial" ]]; then status=partial; failed=1; fi
        if [[ ! -f "${DIRS[$i]}/$stage.done" ]]; then status=failed_or_blocked; failed=1; fi
        printf '%s\t%s\t%s\n' "$batch" "$stage" "$status"
    done
done
exit "$failed"
