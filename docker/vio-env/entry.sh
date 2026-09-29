#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME" /home/user/data/output/logs
action=${1:-doctor}
if (($#)); then shift; fi
log=$(mktemp "/home/user/data/output/logs/${action//[^a-zA-Z0-9_-]/_}-XXXXXX.log")
exec > >(tee "$log") 2>&1
trap 'status=$?; printf "Exit status: %s\n" "$status"' EXIT
printf 'Action: %s\nUTC: %s\n' "$action" "$(date -u +%FT%TZ)"
printf 'Arguments:'; printf ' %q' "$@"; printf '\n'
for repo in /home/user/roboego-hand-vis /home/user/open_vins; do
    printf 'Repository: %s\n' "$repo"
    git -c safe.directory="$repo" -C "$repo" rev-parse HEAD
    git -c safe.directory="$repo" -C "$repo" status --short
done
stage=/home/user/roboego-hand-vis/roboego_hand_vis/vio
build=/home/user/build/openvins
case "$action" in
    doctor)
        python3 -c 'import cv2, numpy, scipy, yaml; import roboego_hand_vis.vio; print("Python dependencies OK; OpenCV", cv2.__version__)'
        cmake --version
        c++ --version
        ;;
    build)
        cmake -S /home/user/open_vins/ov_msckf -B "$build" -G Ninja \
            -DENABLE_ROS=OFF -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
        target=${1:-roboego_openvins_offline}
        cmake --build "$build" --parallel "${BUILD_JOBS:-2}" --target "$target"
        ;;
    prepare|run)
        extra=()
        if [[ "$action" == prepare ]]; then extra+=(--prepare_only); fi
        python3 -m roboego_hand_vis.vio.run \
            --session /home/user/data/input --output_dir /home/user/data/output/run \
            --openvins_root /home/user/open_vins \
            --openvins_binary "$build/roboego_openvins_offline" "${extra[@]}" "$@"
        ;;
    test)
        PYTHONPATH="/home/user/open_vins:${PYTHONPATH:-}" \
            VIO_OFFLINE_BINARY="$build/openvins_offline_native" \
            python3 -m pytest -p no:cacheprovider /home/user/roboego-hand-vis/tests/vio \
            /home/user/open_vins/ov_offline/tests -q "$@"
        ;;
    *) "$action" "$@" ;;
esac
