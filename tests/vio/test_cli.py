from __future__ import annotations

from roboego_hand_vis.vio import cli


def test_cli_overrides_imu_boundary_gap() -> None:
    args = cli.parse_args(
        [
            "--session",
            "/tmp/session",
            "--max_imu_boundary_gap_s",
            "0.3",
        ]
    )

    config = cli.config_from_yaml_and_args(
        {
            "prepare": {"max_imu_boundary_gap_s": 0.1},
            "io": {"output_dir": "{session}/vio"},
            "openvins": {
                "root": "/tmp/open_vins",
                "binary": "{openvins_root}/ov_msckf/build/roboego_openvins_offline",
            },
            "runtime": {"system_python": "/usr/bin/python3"},
        },
        args,
    )

    assert config.max_imu_boundary_gap_s == 0.3
