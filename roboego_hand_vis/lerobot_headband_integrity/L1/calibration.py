"""Check calibration content presence; consumers validate numeric parameters."""
from ..common import result


def check(ctx):
    import yaml

    checks = []
    directory = ctx.root / "calibrations"
    # Stereo calibration is required. Camera/IMU files are only VIO prerequisites.
    for name, level in (("camera-camchain.yaml", "fail"),
                        ("camera_imu-camchain-imucam.yaml", "warning"),
                        ("camera_imu-imu.yaml", "warning")):
        path = directory / name
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            present = isinstance(data, dict) and bool(data)
            detail = None if present else "Calibration must contain a nonempty YAML mapping"
        except (OSError, yaml.YAMLError) as exc:
            present, detail = False, str(exc)
        checks.append(result(f"{name}/content_present", present, level=level, detail=detail))
    return checks
