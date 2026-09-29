#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec /home/user/envs/egohandkit/bin/python "$here/pipeline-output.py" "$@"
