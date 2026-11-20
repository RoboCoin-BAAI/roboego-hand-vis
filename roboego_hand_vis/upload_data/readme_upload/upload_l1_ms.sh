#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

if (( $# > 1 )) || [[ $# == 1 && $1 != --dry-run ]]; then
    printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
    exit 2
fi

data_root=/share/project/liuyou/humancoin/data
python_bin=${HUMANCOIN_PYTHON:-/share/project/liuyou/humancoin/envs/ego/bin/python}
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

folders=()
for root in "$data_root/jianzhi/wait_for_publish/publish" "$data_root/guanglun_depth/publish"; do
    groups=("$root"/l1_*/)
    if (( ${#groups[@]} == 0 )); then
        printf 'No L1 dataset directories in %s\n' "$root" >&2
        exit 1
    fi
    folders+=("${groups[@]}")
done

if (( $# == 0 )); then
    : "${MODELSCOPE_TOKEN:?Export MODELSCOPE_TOKEN before running this script}"
fi

# Keep proxy settings from the calling shell out of the ModelScope upload.
exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY "$python_bin" -m roboego_hand_vis.upload_data.readme_upload.upload2hub \
    --hub ms --namespace HumanCOIN --create-repo --skip-errors \
    --folders "${folders[@]}" "$@"
