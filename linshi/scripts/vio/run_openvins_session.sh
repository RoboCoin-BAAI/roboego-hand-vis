#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENVINS_ROOT="${OPENVINS_ROOT:-/home/user/open_vins}"
OPENVINS_INSTALL="${OPENVINS_INSTALL:-}"
OPENVINS_DEPS="${OPENVINS_DEPS:-/home/user/.local/openvins-deps/usr}"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/bin/python3}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-84}"
export ROS_DOMAIN_ID

session=""
output=""
start_s="0"
duration_s=""
rate="1.0"
downsample="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) session="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --start_s) start_s="$2"; shift 2 ;;
    --duration_s) duration_s="$2"; shift 2 ;;
    --rate) rate="$2"; shift 2 ;;
    --openvins_root) OPENVINS_ROOT="$2"; shift 2 ;;
    --no-downsample) downsample="false"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

OPENVINS_INSTALL="${OPENVINS_INSTALL:-$OPENVINS_ROOT/install_ros2}"

if [[ -z "$session" || -z "$output" ]]; then
  echo "Usage: $0 --session SESSION --output OUTPUT [--duration_s SECONDS] [--rate RATE]" >&2
  exit 2
fi
if [[ ! -f "$OPENVINS_INSTALL/setup.bash" ]]; then
  echo "OpenVINS is not built: $OPENVINS_INSTALL/setup.bash is missing" >&2
  exit 2
fi

mkdir -p "$output"
prepare_args=(
  "$PROJECT_ROOT/scripts/vio/prepare_openvins_session.py"
  --session "$session"
  --output "$output"
)
if [[ "$downsample" == "false" ]]; then
  prepare_args+=(--no-downsample_cameras)
fi
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$SYSTEM_PYTHON" "${prepare_args[@]}"

set +u
source /opt/ros/humble/setup.bash
source "$OPENVINS_INSTALL/setup.bash"
set -u
export LD_LIBRARY_PATH="$OPENVINS_DEPS/lib:$OPENVINS_DEPS/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

config="$output/config/estimator_config.yaml"
manifest="$output/run_manifest.json"
openvins_log="$output/openvins.log"

setsid ros2 run ov_msckf run_subscribe_msckf --ros-args \
  -r __ns:=/ov_msckf \
  -p config_path:="$config" \
  -p save_total_state:=true \
  -p filepath_est:="$output/openvins_state_estimate.txt" \
  -p filepath_std:="$output/openvins_state_deviation.txt" \
  >"$openvins_log" 2>&1 &
openvins_pid=$!

cleanup() {
  if kill -0 "$openvins_pid" 2>/dev/null; then
    kill -INT -- "-$openvins_pid" 2>/dev/null || true
    wait "$openvins_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

replay_args=(
  "$PROJECT_ROOT/scripts/vio/replay_openvins_session.py"
  --manifest "$manifest"
  --start_s "$start_s"
  --rate "$rate"
)
if [[ -n "$duration_s" ]]; then
  replay_args+=(--duration_s "$duration_s")
fi
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$SYSTEM_PYTHON" "${replay_args[@]}"
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$SYSTEM_PYTHON" \
  "$PROJECT_ROOT/scripts/vio/summarize_openvins_run.py" \
  --timing "$output/openvins_timing.txt" \
  --report "$output/trajectory/run_report.json"
