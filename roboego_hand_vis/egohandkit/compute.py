from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EgoHandKitConfig:
    input: Path
    output_root: Path
    egohandkit_root: Path = Path("../EgoHandKit")
    python: str = "python"
    run_script: str = "run.py"
    frontend: str = "canonical"
    observations: str = "observations.jsonl"
    backend: str = "hamer"
    depth_dir: Path | None = None
    prediction: Path | None = None
    gpu: int | None = 0
    batch_size: int | None = 16
    depth_gate: bool = True
    motion_gate: bool = True
    mint_3d_consistency_gate: bool = True
    render_hand_tracking_parquet: bool = True
    hand_tracking_parquet: bool = True
    no_omega_world: bool = True
    extra_args: list[str] = field(default_factory=list)


def _append_optional_path(args: list[str], name: str, value: Path | None) -> None:
    if value is not None:
        args.extend([name, str(value)])


def _append_optional_value(args: list[str], name: str, value: Any) -> None:
    if value is not None and value != "":
        args.extend([name, str(value)])


def build_egohandkit_command(config: EgoHandKitConfig) -> list[str]:
    script = config.egohandkit_root / config.run_script
    args = [
        config.python,
        str(script),
        "--input",
        str(config.input),
        "--output_root",
        str(config.output_root),
        "--frontend",
        config.frontend,
        "--observations",
        config.observations,
        "--backend",
        config.backend,
    ]

    _append_optional_path(args, "--depth_dir", config.depth_dir)
    _append_optional_path(args, "--prediction", config.prediction)
    _append_optional_value(args, "--gpu", config.gpu)
    _append_optional_value(args, "--batch_size", config.batch_size)

    if config.depth_gate:
        args.append("--depth_gate")
    if config.motion_gate:
        args.append("--motion_gate")
    if config.mint_3d_consistency_gate:
        args.append("--mint_3d_consistency_gate")
    if config.hand_tracking_parquet:
        args.append("--hand_tracking_parquet")
    if config.render_hand_tracking_parquet:
        args.append("--render_hand_tracking_parquet")
    if config.no_omega_world:
        args.append("--no_omega_world")

    args.extend(config.extra_args)
    return args


def run_egohandkit(config: EgoHandKitConfig) -> dict[str, Any]:
    command = build_egohandkit_command(config)
    subprocess.run(command, check=True, cwd=str(config.egohandkit_root))
    return {
        "stage": "egohandkit",
        "status": "complete",
        "input": str(config.input),
        "output_root": str(config.output_root),
        "egohandkit_root": str(config.egohandkit_root),
        "command": command,
    }

