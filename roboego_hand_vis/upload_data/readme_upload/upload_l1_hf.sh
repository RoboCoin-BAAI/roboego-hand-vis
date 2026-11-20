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
    : "${HF_TOKEN:?Export HF_TOKEN before running this script}"
    : "${https_proxy:?Export https_proxy before running this script}"
    : "${http_proxy:?Export http_proxy before running this script}"
fi

exec "$python_bin" -m roboego_hand_vis.upload_data.readme_upload.upload2hub \
    --hub hf --namespace HumanCOIN --create-repo --skip-errors \
    --folders "${folders[@]}" "$@"
