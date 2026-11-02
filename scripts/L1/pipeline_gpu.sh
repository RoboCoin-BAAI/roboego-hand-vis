#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Override PIPELINE_PYTHON / PIPELINE_CONFIG on each machine as needed.
PYTHON=${PIPELINE_PYTHON:-/share/project/liuyou/humancoin/envs/ego/bin/python}
exec "$PYTHON" "$HERE/../pipeline/run.py" --config "${PIPELINE_CONFIG:-$HERE/pipeline.yaml}" --role gpu "$@"
