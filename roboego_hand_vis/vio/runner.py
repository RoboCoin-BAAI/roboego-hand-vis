from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenVinsRunConfig:
    session_dir: Path
    output_dir: Path
    openvins_root: Path
    openvins_binary: Path
    system_python: Path
    start_s: float
    duration_s: float | None
    input_mode: str
    downsample_cameras: bool
    stereo_tolerance_s: float
    max_imu_gap_s: float
    max_imu_boundary_gap_s: float
    prepare_only: bool = False


def _quote(value: str | Path | float | int) -> str:
    return shlex.quote(str(value))


def build_openvins_command(config: OpenVinsRunConfig, *, stage_dir: Path) -> str:
    output = config.output_dir
    manifest = output / "run_manifest.json"
    timing = output / "openvins_timing.txt"
    report = output / "trajectory" / "run_report.json"
    py = _quote(config.system_python)
    stage = Path(stage_dir).resolve()

    prepare_parts = [
        py,
        _quote(stage / "prepare_entry.py"),
        "--session",
        _quote(config.session_dir),
        "--output",
        _quote(output),
        "--input_mode",
        _quote(config.input_mode),
        "--stereo_tolerance_s",
        _quote(config.stereo_tolerance_s),
        "--max_imu_gap_s",
        _quote(config.max_imu_gap_s),
        "--max_imu_boundary_gap_s",
        _quote(config.max_imu_boundary_gap_s),
    ]
    if not config.downsample_cameras:
        prepare_parts.append("--no-downsample_cameras")
    commands = [
        "set -euo pipefail",
        f"mkdir -p {_quote(output)}",
        " ".join(prepare_parts),
    ]
    if config.prepare_only:
        return "\n".join(commands)

    offline_parts = [
        py,
        _quote(stage / "offline_entry.py"),
        "--binary",
        _quote(config.openvins_binary),
        "--manifest",
        _quote(manifest),
        "--start_s",
        _quote(config.start_s),
    ]
    if config.duration_s is not None:
        offline_parts.extend(["--duration_s", _quote(config.duration_s)])

    commands.extend(
        [
            " ".join(offline_parts),
            (
                f"if [ -f {_quote(timing)} ] && [ -f {_quote(report)} ]; then "
                f"{py} {_quote(stage / 'summarize.py')} "
                f"--timing {_quote(timing)} "
                f"--report {_quote(report)}; "
                "fi"
            ),
        ]
    )
    return "\n".join(commands)


def run_openvins(config: OpenVinsRunConfig, *, stage_dir: Path | None = None) -> None:
    if not config.openvins_binary.exists() and not config.prepare_only:
        raise FileNotFoundError(
            f"OpenVINS offline runner is not built: {config.openvins_binary} is missing"
        )
    stage = stage_dir or Path(__file__).resolve().parent
    command = build_openvins_command(config, stage_dir=stage)
    subprocess.run(["bash", "-lc", command], check=True)
