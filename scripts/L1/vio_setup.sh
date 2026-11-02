#!/usr/bin/env bash
set -Eeuo pipefail
: "${OPENVINS_ROOT:?Missing OPENVINS_ROOT}"
if [[ ${VIO_INSTALL_DEPS:-1} == 1 ]]; then
    [[ $EUID == 0 ]] || { echo 'VIO dependency installation requires root; preinstall dependencies and set VIO_INSTALL_DEPS=0' >&2; exit 2; }
    export DEBIAN_FRONTEND=noninteractive
    apt_options=(-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)
    apt-get update
    # DEBIAN_FRONTEND alone does not answer dpkg conffile prompts in a Job.
    if ! dpkg --force-confdef --force-confold --configure -a; then
        apt-get "${apt_options[@]}" -f install -y
    fi
    apt-get "${apt_options[@]}" install -y build-essential cmake libeigen3-dev libopencv-dev \
        libopencv-contrib-dev libboost-all-dev libceres-dev
fi
jobs=${VIO_BUILD_JOBS:-2}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || exit 2
OPENVINS_ROOT=$(cd -- "$OPENVINS_ROOT" && pwd -P)
source_dir="$OPENVINS_ROOT/ov_msckf"
build_dir="$source_dir/build-release"
# A shared build may contain absolute paths from the previous machine.
if [[ -f "$build_dir/CMakeCache.txt" ]] &&
   { ! grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$source_dir" "$build_dir/CMakeCache.txt" ||
     ! grep -Fxq "CMAKE_CACHEFILE_DIR:INTERNAL=$build_dir" "$build_dir/CMakeCache.txt" ||
     ! grep -Fxq 'CMAKE_GENERATOR:INTERNAL=Unix Makefiles' "$build_dir/CMakeCache.txt"; }; then
    backup="$build_dir.previous-$(date +%Y%m%d-%H%M%S)-$$"
    echo "Moved CMake cache detected; preserving old build at $backup"
    mv -- "$build_dir" "$backup"
fi
cmake -S "$source_dir" -B "$build_dir" -G 'Unix Makefiles' \
    -DENABLE_ROS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM=/usr/bin/make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$build_dir" --target roboego_openvins_offline --parallel "$jobs"
binary="$build_dir/openvins_offline_native"
[[ -x "$binary" ]]
dependencies=$(ldd "$binary")
printf '%s\n' "$dependencies"
if [[ "$dependencies" == *'not found'* ]]; then
    echo 'OpenVINS still has missing runtime libraries' >&2
    exit 1
fi
echo 'VIO build and runtime library check passed'
