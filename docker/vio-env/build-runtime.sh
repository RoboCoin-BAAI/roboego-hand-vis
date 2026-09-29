#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
project=$(cd "$here/../.." && pwd)
openvins=$(realpath "${OPENVINS_ROOT:-$HOME/open_vins}")
base=${VIO_ENV_IMAGE:-roboego-vio-env:dev}
image=${VIO_RUNTIME_IMAGE:-roboego-vio-runtime:dev}
test -f "$openvins/ov_offline/launcher.py.in"
docker image inspect "$base" >/dev/null
exec docker build --network none -f "$here/Dockerfile.runtime" \
    --build-context "openvins=$openvins" --build-context "project=$project" \
    --build-arg "BASE_IMAGE=$base" \
    --build-arg "PROJECT_REVISION=$(git -C "$project" describe --always --dirty --abbrev=40)" \
    --build-arg "OPENVINS_REVISION=$(git -C "$openvins" describe --always --dirty --abbrev=40)" \
    -t "$image" "$here"
