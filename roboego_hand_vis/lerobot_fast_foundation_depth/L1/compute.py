"""Resolve portable LeRobot inputs, then call the existing GPU inference API."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def episode_inputs(session: Path, episode: Path, camera_names=None) -> dict:
    """Use current dataset paths, even when the dataset moved between machines."""
    camera_names = camera_names or {"left": "head_left_rectified", "right": "head_right_rectified"}
    session = session.resolve()
    episodes = sorted((session / "data").glob("chunk-*/episode_*"))
    prefix = f"{episode.parent.name}_{episode.name}_" if len(episodes) > 1 else ""
    manifest_path = session / "calibrations" / f"{prefix}dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("reference_camera", "left") != "left":
        raise ValueError("Only left-reference stereo depth is supported")
    manifest["reference_camera"] = "left"
    for side in ("left", "right"):
        key = f"observation.images.{camera_names[side]}"
        declared = manifest.get('camera_features', {}).get(side)
        if declared is not None and declared != key:
            raise ValueError(f'Manifest camera mismatch: expected {key}, got {declared}')
        path = session / "videos" / episode.parent.name / key / f"{episode.name}.mp4"
        if not path.is_file():
            raise ValueError(f"Missing rectified video: {path}; run stereo rectification first")
        manifest["videos"][side]["path"] = str(path)
        intrinsics = manifest["intrinsics"][side]
        if (not all(math.isfinite(float(intrinsics[k])) for k in ("fx", "fy", "cx", "cy"))
                or float(intrinsics["fx"]) <= 0 or float(intrinsics["fy"]) <= 0):
            raise ValueError("Invalid rectified intrinsics")
    baseline = float(manifest["extrinsics"]["baseline_m"])
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("Stereo baseline must be positive and in meters")
    timestamps = episode / f"observation.images.{camera_names['left']}.parquet"
    if not timestamps.is_file():
        raise ValueError(f"Missing rectified left timestamp feature: {timestamps}")
    manifest["source_episode"] = str(episode)
    mapping = session / "calibrations" / f"{prefix}frame_mapping.parquet"
    manifest["frame_mapping"] = str(mapping) if mapping.is_file() else None
    return dict(manifest=manifest, source_manifest=manifest_path, timestamps=timestamps,
                output=session / "depth" / episode.parent.name / episode.name)


def plan_session(session, camera_names=None):
    session = Path(session).resolve()
    episodes = sorted((session / "data").glob("chunk-*/episode_*"))
    if not episodes:
        raise ValueError("No LeRobot episodes found")
    plans = []
    for episode in episodes:
        inputs = episode_inputs(session, episode, camera_names=camera_names)
        plans.append(dict(episode=str(episode), manifest=str(inputs["source_manifest"]),
            output=str(inputs["output"]), left_video=inputs["manifest"]["videos"]["left"]["path"],
            right_video=inputs["manifest"]["videos"]["right"]["path"]))
    return plans


def run_session(session, *, config=DEFAULT_CONFIG, model_root=None, ckpt=None,
                num_frames="auto", start=0, save_vis=None, preview_every_n_frames=None, camera_names=None):
    import yaml
    from roboego_hand_vis.fast_foundation_depth.cli import config_from_yaml
    from roboego_hand_vis.fast_foundation_depth.compute import run_fast_foundation_depth

    session = Path(session).resolve()
    settings = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Depth config must be a YAML mapping")
    paths = settings.setdefault("paths", {})
    for name, override in (("fast_foundation_root", model_root), ("ckpt", ckpt)):
        value = override or paths.get(name)
        if not value:
            raise ValueError(f"Missing model path: {name}")
        path = Path(value).expanduser()
        paths[name] = str((Path(config).resolve().parent / path).resolve() if not path.is_absolute() else path)
    if not Path(paths["fast_foundation_root"]).is_dir():
        raise ValueError("Fast-FoundationStereo source directory does not exist")
    checkpoint = Path(paths["ckpt"])
    if not checkpoint.is_file() or not checkpoint.with_name("cfg.yaml").is_file():
        raise ValueError("Model checkpoint and its adjacent cfg.yaml must exist")
    settings.setdefault("parameters", {}).update(num_frames=num_frames, start=start, overwrite=True)
    for key, value in (("save_vis", save_vis), ("preview_every_n_frames", preview_every_n_frames)):
        if value is not None:
            settings["parameters"][key] = value
    # Sequential reader in the existing backend must process every requested
    # frame. Always recompute the range instead of skipping existing files.
    results = []
    for plan in plan_session(session, camera_names=camera_names):
        inputs = episode_inputs(session, Path(plan["episode"]), camera_names=camera_names)
        from roboego_hand_vis.common.stereo_quality import require_stereo_quality
        require_stereo_quality(session, Path(plan['episode']), inputs['manifest']['videos']['left']['path'],
                               inputs['manifest']['videos']['right']['path'])
        out = inputs["output"]
        if out.is_symlink() or not out.resolve().is_relative_to(session):
            raise ValueError("Depth output escapes session")
        out.mkdir(parents=True, exist_ok=True)
        # Existing loader expects a JSON path with absolute video paths.
        with tempfile.TemporaryDirectory(prefix="lerobot-depth-") as tmp:
            runtime_manifest = Path(tmp) / "dataset_manifest.json"
            runtime_manifest.write_text(json.dumps(inputs["manifest"]), encoding="utf-8")
            cfg = config_from_yaml(settings, manifest=runtime_manifest, output_dir=out)
            result = run_fast_foundation_depth(cfg)
        shutil.copy2(inputs["timestamps"], out / "source_timestamps.parquet")
        provenance = dict(stage="lerobot_fast_foundation_depth", source_episode=plan["episode"],
            source_manifest=str(inputs["source_manifest"]), camera_feature=inputs["timestamps"].stem,
            start=start, requested_num_frames=num_frames, processed_frames=result["processed_frames"],
            timestamp_lookup="source_timestamps.parquet row indexed by original frame number",
            synchronization_checked=False)
        (out / "lerobot_depth.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        # Replace the temporary manifest reference in backend metadata.
        metadata = out / "fast_foundation_stereo_video_meta.json"
        if metadata.is_file():
            data = json.loads(metadata.read_text(encoding="utf-8"))
            data["manifest_path"] = str(inputs["source_manifest"])
            metadata.write_text(json.dumps(data, indent=2), encoding="utf-8")
        results.append(dict(episode=plan["episode"], output=str(out), processed_frame_count=result["processed_frame_count"]))
    return results
