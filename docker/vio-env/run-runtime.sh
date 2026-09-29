#!/usr/bin/env bash
set -euo pipefail
: "${INPUT_DIR:?Set INPUT_DIR to the session directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh result directory}"
image=${VIO_RUNTIME_IMAGE:-roboego-vio-runtime:dev}
input=$(realpath "$INPUT_DIR")
test -d "$input"
mkdir -p "$OUTPUT_DIR"
output=$(realpath "$OUTPUT_DIR")
destination=${INPUT_CONTAINER_PATH:-/data/input}
[[ "$destination" == /* && "$destination" != / && "$destination" != /data && "$destination" != /data/output && "$destination" != /data/output/* && "$destination" != /opt && "$destination" != /opt/* ]] || {
    printf 'Invalid input mount destination: %s\n' "$destination" >&2; exit 1;
}
for path in "$input" "$output" "$destination"; do
    [[ "$path" != *,* ]] || { printf 'Mount paths cannot contain commas\n' >&2; exit 1; }
done
id=$(docker image inspect --format '{{.Id}}' "$image")
exec docker run --rm --init --network none --user "$(id -u):$(id -g)" \
    -e "VIO_IMAGE_ID=$id" \
    -e "INPUT_CONTAINER_PATH=$destination" \
    --mount "type=bind,src=$input,dst=$destination,readonly" \
    --mount "type=bind,src=$output,dst=/data/output" \
    "$id" "$@"
