#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME"
action=${1:-doctor}
if (($#)); then shift; fi
case "$action" in
    doctor)
        python3 -c 'import ctypes, cv2, numpy, yaml; import roboego_hand_vis.vio, ov_offline.run; ctypes.CDLL("/opt/open_vins/ov_msckf/build/libov_msckf_lib.so"); print("Bundled VIO runtime: Python imports and native library OK")'
        /opt/open_vins/ov_msckf/build/roboego_openvins_offline --help
        ;;
    test)
        VIO_OFFLINE_BINARY=/opt/open_vins/ov_msckf/build/openvins_offline_native \
            python3 -m pytest -p no:cacheprovider /opt/open_vins/ov_offline/tests \
            /opt/roboego-hand-vis/tests/vio /usr/local/lib/vio-runtime/tests -q "$@"
        ;;
    run)
        exec python3 /usr/local/lib/vio-runtime/runtime-output.py "$@"
        ;;
    python|python3)
        if [[ "${1:-}" == -m && "${2:-}" == roboego_hand_vis.vio.run ]]; then
            shift 2
            exec python3 /usr/local/lib/vio-runtime/runtime-output.py "$@"
        fi
        exec "$action" "$@"
        ;;
    prepare)
        mkdir -p /data/output/logs
        log=$(mktemp /data/output/logs/vio-XXXXXX.log)
        exec > >(tee "$log") 2>&1
        extra=()
        if [[ "$action" == prepare ]]; then extra+=(--prepare_only); fi
        python3 -m roboego_hand_vis.vio.run \
            --session "${INPUT_CONTAINER_PATH:-/data/input}" --output_dir /data/output/run \
            --openvins_root /opt/open_vins \
            --openvins_binary /opt/open_vins/ov_msckf/build/roboego_openvins_offline \
            "${extra[@]}" "$@"
        ;;
    *) exec "$action" "$@" ;;
esac
