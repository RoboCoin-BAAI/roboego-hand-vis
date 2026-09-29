#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OPENVINS_ROOT=${OPENVINS_ROOT:-$HOME/open_vins}
: "${INPUT_DIR:?Set INPUT_DIR to a raw session containing videos, timestamps, imu and calibrations}"
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/tmp/vio_container/output}
BUILD_DIR=${BUILD_DIR:-$PROJECT_ROOT/tmp/vio_container/build}
IMAGE=${VIO_ENV_IMAGE:-roboego-vio-env:dev}
for path in "$PROJECT_ROOT/roboego_hand_vis/vio/run.py" "$OPENVINS_ROOT/ov_msckf/CMakeLists.txt" "$INPUT_DIR"; do
    [[ -e "$path" ]] || { printf 'Missing mount source: %s\n' "$path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR" "$BUILD_DIR"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
OPENVINS_ROOT=$(realpath "$OPENVINS_ROOT")
INPUT_DIR=$(realpath "$INPUT_DIR")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
BUILD_DIR=$(realpath "$BUILD_DIR")
for path in "$PROJECT_ROOT" "$OPENVINS_ROOT" "$INPUT_DIR" "$OUTPUT_DIR" "$BUILD_DIR"; do
    [[ "$path" != *,* ]] || { printf 'Docker mount paths cannot contain commas: %s\n' "$path" >&2; exit 1; }
done
image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
mkdir -p "$OUTPUT_DIR/logs"
printf '%s\n' "$image_id" > "$OUTPUT_DIR/logs/image-id.txt"
exec docker run --rm --init --network none --entrypoint /bin/bash --user "$(id -u):$(id -g)" \
    -e BUILD_JOBS="${BUILD_JOBS:-2}" \
    --mount "type=bind,src=$PROJECT_ROOT,dst=/home/user/roboego-hand-vis,readonly" \
    --mount "type=bind,src=$OPENVINS_ROOT,dst=/home/user/open_vins,readonly" \
    --mount "type=bind,src=$INPUT_DIR,dst=/home/user/data/input,readonly" \
    --mount "type=bind,src=$OUTPUT_DIR,dst=/home/user/data/output" \
    --mount "type=bind,src=$BUILD_DIR,dst=/home/user/build" \
    "$image_id" /home/user/roboego-hand-vis/docker/vio-env/entry.sh "$@"
