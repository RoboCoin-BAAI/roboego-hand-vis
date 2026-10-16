#!/usr/bin/env bash
# bash vio-run.sh SESSION EMPTY_OUTPUT run [project VIO CLI arguments...]
set -euo pipefail
fail() { echo "$*" >&2; exit 1; }
(($# >= 3)) || fail 'Usage: vio-run.sh SESSION EMPTY_OUTPUT run [VIO arguments...]'
R=$(realpath -e "${VIO_ROOTFS:-/home/humancoindata/vio-rootfs}")
IN=$(realpath -e "$1")
[[ -d "$R" && "$R" != / && -d "$IN" && "$IN" != / ]] || fail 'Invalid rootfs or session'
case "$IN/" in "$R/"*) fail 'Session must be outside rootfs';; esac
case "$R/" in "$IN/"*) fail 'Rootfs must be outside session';; esac
OUT=$2
shift 2
[[ "$1" == run ]] || fail 'Only run is supported; use the project CLI for prepare'
shift
# Paths belong to the transport layer. All algorithm options pass through unchanged.
for arg in "$@"; do
  case "$arg" in
    --session|--session=*|--output_dir|--output_dir=*|--prepare_only)
      fail "Transport owns session/output and requires a full run: $arg";;
  esac
done
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
case "$OUT/" in "$R/"*) fail 'Output must be outside rootfs';; esac
# Distinct from the database's script lock; manual and database runs share this lock.
exec 9>"$R/.vio-run.lock"
flock --exclusive 9
[[ -z "$(ls -A "$OUT")" ]] || fail "Output directory must be empty: $OUT"
STAGED="$R$IN"
[[ "$(realpath -m "$STAGED")" == "$STAGED" ]] || fail 'Staged input crosses a symlink'
[[ ! -e "$STAGED" && ! -L "$STAGED" ]] || fail "Staged input already exists; inspect it first: $STAGED"
[[ "$(realpath -m "$R/data/output")" == "$R/data/output" ]] || fail 'Rootfs output crosses a symlink'
mkdir -p "$R/data/output" "$R/tmp/vio-home" "$R/dev" "$(dirname "$STAGED")"
[[ -z "$(ls -A "$R/data/output")" ]] || fail "Rootfs output is not empty; inspect it first: $R/data/output"
# Same-filesystem hard links, as in the original deployment. Inputs must remain read-only.
cp -al "$IN" "$STAGED"
for d in null:1:3 zero:1:5 random:1:8 urandom:1:9; do
  IFS=: read -r n a b <<< "$d"
  [[ -e "$R/dev/$n" ]] || mknod -m 666 "$R/dev/$n" c "$a" "$b"
done
set +e
chroot "$R" /usr/bin/env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HOME=/tmp/vio-home PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  PYTHONPATH=/opt/roboego-hand-vis:/opt/open_vins \
  LD_LIBRARY_PATH=/opt/open_vins/ov_msckf/build \
  OPENVINS_TIME_OFFSET_MODE="${OPENVINS_TIME_OFFSET_MODE:-estimate}" \
  OPENVINS_TIMING_BOOTSTRAP="${OPENVINS_TIMING_BOOTSTRAP:-auto}" \
  /usr/bin/python3 -m roboego_hand_vis.vio.run "$@" \
  --session "$IN" --output_dir /data/output \
  --openvins_root /opt/open_vins \
  --openvins_binary /opt/open_vins/ov_msckf/build/roboego_openvins_offline \
  > "$R/data/output/console.log" 2>&1
rc=$?
set -e
cat "$R/data/output/console.log"
if ((rc != 0)); then
  cp "$R/data/output/console.log" "$R/data/output/error.log"
fi
# Preserve all intermediate files for diagnosis. If copy fails, keep rootfs files too.
cp -a "$R/data/output/." "$OUT/"
rm -rf -- "$STAGED" "$R/data/output"
echo "VIO exit=$rc; output=$OUT"
exit "$rc"
