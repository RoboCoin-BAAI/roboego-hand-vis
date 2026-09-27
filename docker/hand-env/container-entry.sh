#!/usr/bin/env bash
set -euo pipefail
case "${1:-doctor}" in
  doctor)
    /home/user/envs/mint/bin/python /usr/local/lib/hand-env/smoke.py mint
    /home/user/envs/egohandkit/bin/python /usr/local/lib/hand-env/smoke.py egohandkit
    ;;
  mint)
    shift
    cd /home/user/wuji-ego-mint
    exec /home/user/envs/mint/bin/python -m mint "$@"
    ;;
  export)
    shift
    cd /home/user/wuji-ego-mint
    exec /home/user/envs/mint/bin/python scripts/export_egohandkit_cache.py "$@"
    ;;
  egohandkit)
    shift
    cd /home/user/EgoHandKit
    exec /home/user/envs/egohandkit/bin/python run.py "$@"
    ;;
  *) exec "$@" ;;
esac
