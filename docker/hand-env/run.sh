#!/usr/bin/env bash
set -euo pipefail
# Host paths are configurable; container paths are deliberately stable.
IMAGE=${HAND_ENV_IMAGE:-roboego-hand-env:dev}
MINT_ROOT=${MINT_ROOT:-$HOME/wuji-ego-mint}
EGO_ROOT=${EGO_ROOT:-$HOME/EgoHandKit}
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
INPUT_DIR=${INPUT_DIR:-$PROJECT_ROOT/tmp/container_validation}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/tmp/container_validation/container_output}
for path in "$MINT_ROOT/mint" "$EGO_ROOT/run.py" "$PROJECT_ROOT/roboego_hand_vis" "$INPUT_DIR"; do
  [[ -e "$path" ]] || { printf 'Missing mount source: %s\n' "$path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
MINT_ROOT=$(realpath "$MINT_ROOT")
EGO_ROOT=$(realpath "$EGO_ROOT")
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
INPUT_DIR=$(realpath "$INPUT_DIR")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
# Copy-up once into an image-specific named volume. Nested read-only mounts
# override host _DATA/MANO symlinks without modifying either source repository.
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
MODEL_VOLUME="roboego-hand-models-${IMAGE_ID#sha256:}"
docker volume create "$MODEL_VOLUME" >/dev/null
docker run --rm --network none --entrypoint /bin/true \
  --mount "type=volume,src=$MODEL_VOLUME,dst=/home/user/model" "$IMAGE"
WORK_DIR=$(mktemp -d "$OUTPUT_DIR/.frames-XXXXXX")
exec docker run --rm --init --network none --gpus "${HAND_GPUS:-all}" --shm-size 2g \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$MINT_ROOT,dst=/home/user/wuji-ego-mint,readonly" \
  --mount "type=bind,src=$EGO_ROOT,dst=/home/user/EgoHandKit,readonly" \
  --mount "type=bind,src=$PROJECT_ROOT,dst=/home/user/roboego-hand-vis,readonly" \
  --mount "type=bind,src=$INPUT_DIR,dst=/home/user/data/input,readonly" \
  --mount "type=bind,src=$OUTPUT_DIR,dst=/home/user/data/output" \
  --mount "type=bind,src=$WORK_DIR,dst=/home/user/EgoHandKit/test_data" \
  --mount "type=volume,src=$MODEL_VOLUME,dst=/home/user/model,readonly" \
  --mount "type=volume,src=$MODEL_VOLUME,volume-subpath=egohandkit,dst=/home/user/EgoHandKit/_DATA,readonly" \
  --mount "type=volume,src=$MODEL_VOLUME,volume-subpath=mano,dst=/home/user/wuji-ego-mint/assets/mano,readonly" \
  "$IMAGE" "$@"
