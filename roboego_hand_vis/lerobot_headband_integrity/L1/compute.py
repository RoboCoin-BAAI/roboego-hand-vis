"""Modality registry and unified report; add modalities here to extend checks."""
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from . import audio, calibration, imu, timestamps, video, stereo
from ..common import CAMERAS, Context, guarded, result

MODALITIES = {"timestamps": timestamps.check, "video": video.check,
              "imu": imu.check, "audio": audio.check, "calibration": calibration.check,
              "stereo_first_frame": stereo.check}


def build_report(session, validate_vio_calibration=True, **options):
    checks = []
    try:
        ctx = Context(Path(session), **options)
        if validate_vio_calibration:
            from .calibration_repair import complete_vio_file, FILES
            for name in FILES:
                valid, detail = complete_vio_file(ctx.root / 'calibrations' / name)
                checks.append(result(f'calibration/{name}/repair_structure', valid, detail=detail))
        features = ctx.info.get("features", {})
        checks.append(result("dataset/features", all(f"observation.images.{c}" in features for c in CAMERAS)
                             and "observation.imu" in features))
        checks.append(result("dataset/episode_count", ctx.info.get("total_episodes") == len(ctx.episodes),
                             actual_episodes=len(ctx.episodes)))
        for modality, check in MODALITIES.items():
            tick=time.monotonic()
            print(f'[qc-progress] {ctx.root.name} {modality} start',flush=True)
            for item in guarded("readable", lambda: check(ctx)):
                item["name"] = f"{modality}/{item['name']}"
                checks.append(item)
            print(f'[qc-progress] {ctx.root.name} {modality} finished elapsed_s={time.monotonic()-tick:.1f}',flush=True)
    except Exception as exc:
        checks.append(result("dataset/readable", False, detail=f"{type(exc).__name__}: {exc}"))
    failed = [c["name"] for c in checks if not c["passed"] and c["level"] == "fail"]
    warnings = [c["name"] for c in checks if not c["passed"] and c["level"] == "warning"]
    return dict(schema_version="1.0", stage="lerobot_headband_integrity",
                source_session=str(Path(session).resolve()), created_at=datetime.now(timezone.utc).isoformat(),
                passed=not failed, gate=dict(passed=not failed, failed_checks=failed, warnings=warnings), checks=checks)


def export_report(session, **options):
    report = build_report(session, **options)
    receipt = Path(session) / 'annotation/camera_calibration_repair.json'
    if receipt.is_file():
        try:
            report['camera_calibration_repair'] = json.loads(receipt.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    path = Path(session) / "annotation/quality_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return dict(session=str(Path(session)), report=str(path), passed=report["passed"],
                failed_checks=report["gate"]["failed_checks"], warnings=report["gate"]["warnings"])
