"""Rectify the L1 front stereo pair into a separate derived-data directory."""
from __future__ import annotations

import json
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import shutil
from uuid import uuid4

from roboego_hand_vis.lerobot_headband_integrity.common import Context


def load_geometry(path):
    import numpy as np
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return geometry_from_mapping(data)


def geometry_from_mapping(data):
    """Validate an in-memory Kalibr mapping, also used by the L3 JSONL adapter."""
    import numpy as np

    if "head_left" in data and "head_right" in data:
        left, right = data["head_left"], data["head_right"]
        reference = right.get("extrinsics_reference_camera")
    else:
        # Original Kalibr keys are also supported; derive positions from topics.
        keys = {}
        for key, entry in data.items():
            topic = str(entry.get("rostopic", "")).split("/")
            if len(topic) > 2 and topic[2] in ("left", "right"):
                if topic[2] in keys:
                    raise ValueError("Ambiguous calibration camera topics")
                keys[topic[2]] = key
        left_key, right_key = keys["left"], keys["right"]
        left, right = data[left_key], data[right_key]
        reference = ("head_left" if left_key.startswith("cam") and right_key.startswith("cam")
                     and int(right_key[3:]) == int(left_key[3:]) + 1 else None)
    from roboego_hand_vis.lerobot_headband_integrity.camera_calibration import camera_error
    for camera in (left, right):
        pair = (camera.get('camera_model'), camera.get('distortion_model'))
        if pair not in (('pinhole', 'equidistant'), ('ds', 'none')):
            raise ValueError('Only pinhole + equidistant or ds + none calibration is supported')
        error = camera_error(camera)
        if error:
            raise ValueError(error)
    if left['camera_model'] != right['camera_model']:
        raise ValueError('Mixed stereo camera models are not supported')
    if left["resolution"] != right["resolution"]:
        raise ValueError("Stereo calibration resolutions differ")
    if reference != "head_left":
        raise ValueError("Right T_cn_cnm1 must explicitly reference head_left")
    transform = np.asarray(right["T_cn_cnm1"], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)):
        raise ValueError("Invalid left-to-right transform")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4)
            or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-4)):
        raise ValueError("Extrinsic rotation is not a proper rotation")
    if np.linalg.norm(transform[:3, 3]) <= 1e-9:
        raise ValueError("Stereo baseline is zero")
    return left, right, transform


def crop_roi(actual, calibrated, allow_center_crop=True):
    w, h = actual
    cw, ch = calibrated
    if (w, h) != (cw, ch) and not allow_center_crop:
        raise ValueError(f"Video {actual} differs from calibration {calibrated}; confirm crop with --center-crop-calibration")
    if w < cw or h < ch:
        raise ValueError("Video is smaller than calibration; cannot infer a resize")
    x, y = (w - cw) // 2, (h - ch) // 2
    return (x, y, x + cw, y + ch)


class Encoder:
    def __init__(self, path, size, fps, ffmpeg, *, preset="medium", threads=2, log_path=None):
        w, h = size
        path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(log_path) if log_path is not None else path.with_suffix(".ffmpeg.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("wb")
        self.process = None
        try:
            self.process = subprocess.Popen([ffmpeg, "-nostdin", "-v", "error", "-f", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", preset, "-threads", str(threads), "-crf", "18", "-bf", "0", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def close(self):
        try:
            if self.process.stdin and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            code = self.process.wait()
            if code:
                raise RuntimeError(f"ffmpeg exited with code {code}; see encoder log")
            self.log.close()
            Path(self.log.name).unlink(missing_ok=True)
        finally:
            self.log.close()


def crop_output_maps(maps, params, size):
    """Use the original rectifier's valid-ROI heuristic and update geometry."""
    from roboego_hand_vis.stereo_rectification.rectify_video import (
        compute_valid_crop_roi, intersect_roi, crop_remap_maps,
    )
    rois = [compute_valid_crop_roi(*pair, *size) for pair in maps]
    x0, y0, x1, y1 = intersect_roi(*rois)
    width, height = x1 - x0, y1 - y0
    width -= width % 2
    height -= height % 2
    if width <= 0 or height <= 0:
        raise ValueError("Stereo valid regions have no usable common rectangle")
    roi = (x0, y0, x0 + width, y0 + height)
    maps = [crop_remap_maps(*pair, roi) for pair in maps]
    params = {k: v.copy() for k, v in params.items()}
    for key in ("P1", "P2"):
        params[key][0, 2] -= x0
        params[key][1, 2] -= y0
    params["Q"][0, 3] += x0
    params["Q"][1, 3] += y0
    return maps, params, (width, height), roi


def move_published_file(source, target):
    import errno
    try:
        source.rename(target)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.move(str(source), str(target))


def publish_files(staging, root, staging_base=None):
    """Publish staged features and metadata, restoring old files on failure."""
    files = sorted(p for p in staging.rglob("*") if p.is_file())
    # Publish info last, after its feature files are in place.
    files.sort(key=lambda p: p.relative_to(staging).as_posix() == "meta/info.json")
    targets = [(p, root / p.relative_to(staging)) for p in files]
    for _, target in targets:
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Output path escapes dataset: {target}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Output is not a regular file: {target}")
    backups, published = [], []
    try:
        for source, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = staging / ".previous" / target.relative_to(root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                move_published_file(target, backup)
                backups.append((backup, target))
            move_published_file(source, target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            target.unlink()
        for backup, target in reversed(backups):
            move_published_file(backup, target)
        raise
    cleanup_base = Path(staging_base).resolve() if staging_base is not None else root.resolve()
    if staging.is_symlink() or not staging.resolve().is_relative_to(cleanup_base):
        raise ValueError("Refusing to clean staging outside dataset")
    try:
        shutil.rmtree(staging)
    except OSError as exc:
        print(f"Output published; temporary files retained at {staging}: {exc}", flush=True)


def rectify_episode(ctx, episode, *, balance=0., ffmpeg="ffmpeg",
                    center_crop_calibration=True, dry_run=False, skip_existing=True,
                    camera_names=None, geometry=None, calibration_source=None, work_root=None, write_frame_mapping=True, full_decode_validation=True):
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    from roboego_hand_vis.stereo_rectification.equidistant import EquidistantCameraModel
    from roboego_hand_vis.stereo_rectification.double_sphere import DoubleSphereCameraModel

    camera_names = camera_names or {"left": "head_left", "right": "head_right"}
    left, right, transform = (geometry if geometry is not None else
                              load_geometry(ctx.root / "calibrations/camera-camchain.yaml"))
    output = ctx.root
    staging_base = Path(work_root).resolve() if work_root is not None else ctx.root
    if work_root is not None and (staging_base.is_relative_to(ctx.root) or ctx.root.is_relative_to(staging_base)):
        raise ValueError("Runtime work directory must be separate from dataset")
    cal_relative = Path("calibrations")
    cal_output = ctx.root / cal_relative
    prefix = f"{episode.parent.name}_{episode.name}_" if len(ctx.episodes) > 1 else ""
    mapping_name = f"{prefix}frame_mapping.parquet"
    manifest_name = f"{prefix}dataset_manifest.json"
    params_name = f"{prefix}rectify_params.yaml"
    video_relative = {side: Path("videos") / episode.parent.name /
                      f"observation.images.{camera_names[side]}_rectified" / f"{episode.name}.mp4"
                      for side in ("left", "right")}
    timestamp_relative = {side: episode.relative_to(ctx.root) /
                          f"observation.images.{camera_names[side]}_rectified.parquet"
                          for side in ("left", "right")}
    captures, encoders = [], []
    staging = None
    try:
        crops = []
        for camera in (camera_names["left"], camera_names["right"]):
            cap = cv2.VideoCapture(str(ctx.video(episode, camera)))
            captures.append(cap)
            if not cap.isOpened():
                raise ValueError(f"Cannot open {camera} video")
            actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            crops.append(crop_roi(actual, tuple(left["resolution"]), center_crop_calibration))
        plan = dict(output=str(output), calibration_output=str(cal_output),
                    videos={side: str(ctx.root / path) for side, path in video_relative.items()},
                    input_crops=crops, frame_policy="source_order")
        if dry_run:
            return plan
        if left['camera_model'] == 'ds':
            # Kalibr order: xi, alpha, fx, fy, cx, cy. Constructor uses focal first.
            models = [DoubleSphereCameraModel(*c['intrinsics'][2:], *c['intrinsics'][:2]) for c in (left, right)]
        else:
            models = [EquidistantCameraModel(*c["intrinsics"], *c["distortion_coeffs"]) for c in (left, right)]
        size = tuple(left["resolution"])
        params = type(models[0]).stereo_rectify(*models, size, transform[:3, :3], transform[:3, 3], balance=balance)
        p1, p2 = params["P1"], params["P2"]
        # Existing depth code assumes positive horizontal disparity and fx*B/d.
        if p1[0, 0] <= 0 or p2[0, 3] >= 0 or abs(p2[1, 3]) > 1e-6:
            raise ValueError("Rectification is not positive-disparity horizontal stereo")
        maps = [m.generate_stereo_remap(params[f"R{k}"], params[f"P{k}"], *size) for k, m in enumerate(models, 1)]
        maps, params, size, output_roi = crop_output_maps(maps, params, size)
        p1, p2 = params["P1"], params["P2"]
        plan.update(output_resolution=list(size), crop_roi=list(output_roi))
        if skip_existing:
            from .reuse import validate_existing
            try:
                manifest, counts = validate_existing(ctx, episode,
                    manifest_path=cal_output / manifest_name, params_path=cal_output / params_name,
                    mapping_path=(cal_output / mapping_name if write_frame_mapping else None), video_relative=video_relative,
                    timestamp_relative=timestamp_relative, params=params, size=size,
                    crops=crops, output_roi=output_roi, transform=transform, balance=balance,
                    camera_names=camera_names, calibration_source=calibration_source,
                    full_decode_validation=full_decode_validation)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError, cv2.error) as exc:
                print(f"[recompute] {ctx.root.name}/{episode.name}: {exc}", flush=True)
            else:
                old = json.loads((cal_output / manifest_name).read_text(encoding='utf-8'))
                if old != manifest:
                    staging = staging_base / f'.rectification.partial-{uuid4().hex}'
                    (staging / cal_relative).mkdir(parents=True)
                    (staging / cal_relative / manifest_name).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                    publish_files(staging, ctx.root, staging_base=staging_base)
                    staging = None
                print(f"[skip-existing] {ctx.root.name}/{episode.name}: validated rectification", flush=True)
                return dict(**plan, frame_counts=counts, skipped_existing=True)
        staging = staging_base / f".rectification.partial-{uuid4().hex}"
        staging.mkdir(parents=True)
        counts = {}
        for side, cap, crop, maps_side in zip(("left", "right"), captures, crops, maps):
            encoder = Encoder(staging / video_relative[side], size, ctx.fps(camera_names[side]), ffmpeg,
                              **({"log_path": staging_base / "logs" / f"{prefix}{episode.name}_{side}.ffmpeg.log"} if work_root is not None else {}))
            encoders.append(encoder)
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                x0, y0, x1, y1 = crop
                encoder.write(cv2.remap(frame[y0:y1, x0:x1], *maps_side, cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT))
                count += 1
                if count == 1 or count % 300 == 0:
                    print(f"[{ctx.root.name}/{episode.name}/{side}] {count}", flush=True)
            encoder.close()
            encoders.remove(encoder)
            if count == 0:
                raise ValueError(f"{side}: no frames could be read")
            counts[side] = count
            # Preserve acquisition timestamps verbatim when available; they do
            # not control geometry processing or gate frame output.
            timestamps = episode / f"observation.images.{camera_names[side]}.parquet"
            if not timestamps.is_file():
                raise ValueError(f"Missing timestamp feature required for LeRobot output: {timestamps}")
            destination = staging / timestamp_relative[side]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(timestamps, destination)
        (staging / cal_relative).mkdir(parents=True, exist_ok=True)
        length = max(counts.values())
        if write_frame_mapping:
            pq.write_table(pa.table(dict(frame_index=list(range(length)),
                left_source_index=[i if i < counts["left"] else None for i in range(length)],
                right_source_index=[i if i < counts["right"] else None for i in range(length)])), staging / cal_relative / mapping_name)
        plan["frame_counts"] = counts
        intrinsics = {side: dict(fx=float(p[0, 0]), fy=float(p[1, 1]), cx=float(p[0, 2]), cy=float(p[1, 2]))
                      for side, p in (("left", p1), ("right", p2))}
        manifest = dict(dataset_name=ctx.root.name, sequence_name=episode.name, reference_camera="left",
            videos={side: dict(path=str(ctx.root / video_relative[side]), fps=ctx.fps(camera_names[side]), frame_count=counts[side], width=size[0], height=size[1]) for side in ("left", "right")},
            intrinsics=intrinsics, extrinsics=dict(baseline_m=float(-p2[0, 3] / p2[0, 0])),
            frame_mapping=str(cal_output / mapping_name), source_episode=str(episode),
            stereo_pair=[camera_names["left"], camera_names["right"]], frame_policy="source_order", synchronization_checked=False)
        if not write_frame_mapping:
            manifest.pop("frame_mapping", None)
        if calibration_source is not None:
            manifest["calibration_source"] = calibration_source
        manifest["camera_features"] = {side: f"observation.images.{camera_names[side]}_rectified" for side in ("left", "right")}
        (staging / cal_relative / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (staging / cal_relative / params_name).write_text(yaml.safe_dump(dict(
            **{k: v.tolist() for k, v in params.items()}, balance=balance, resolution=list(size),
            input_crops=[list(c) for c in crops], frame_policy="source_order",
            crop_roi=dict(zip(("x_min", "y_min", "x_max", "y_max"), output_roi)),
            cameras={f"{camera_names[side]}_rectified": dict(feature_key=f"observation.images.{camera_names[side]}_rectified",
                camera_model="pinhole", distortion_model="none", distortion_coeffs=[],
                intrinsics=[intrinsics[side][k] for k in ("fx", "fy", "cx", "cy")], resolution=list(size))
                for side in ("left", "right")},
            left_to_right=transform.tolist(), source_camera_features=[f"observation.images.{camera_names[side]}" for side in ("left", "right")])), encoding="utf-8")
        # Register new cameras without touching the original features/index.
        info = json.loads((ctx.root / "meta/info.json").read_text(encoding="utf-8"))
        for side in ("left", "right"):
            key = f"observation.images.{camera_names[side]}_rectified"
            spec = deepcopy(info["features"][f"observation.images.{camera_names[side]}"])
            spec.update(dtype="video", shape=[size[1], size[0], 3], names=["h", "w", "c"])
            spec.setdefault("info", {}).update({"video.width": size[0], "video.height": size[1],
                "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.fps": ctx.fps(camera_names[side]),
                "video.channels": 3, "video.is_depth_map": False, "has_audio": False})
            # LeRobot feature dimensions are dataset-wide, even across episodes.
            others = [p for p in (ctx.root / "videos").glob(f"chunk-*/{key}/episode_*.mp4")
                      if p != ctx.root / video_relative[side]]
            if others and info["features"].get(key, {}).get("shape") != spec["shape"]:
                raise ValueError("Rectified resolutions differ between episodes")
            info["features"][key] = spec
        video_keys = {k for k, v in info["features"].items() if v.get("dtype") == "video"}
        existing = {p.relative_to(ctx.root).as_posix() for p in (ctx.root / "videos").glob("chunk-*/*/episode_*.mp4")
                    if p.parent.name in video_keys}
        info["total_videos"] = len(existing | {p.as_posix() for p in video_relative.values()})
        (staging / "meta").mkdir()
        (staging / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        publish_files(staging, ctx.root, staging_base=staging_base)
        ctx.info = info
        return plan
    finally:
        for cap in captures:
            cap.release()
        for encoder in encoders:
            try:
                encoder.close()
            except Exception:
                pass
        if staging is not None and staging.exists():
            print(f"Incomplete output retained: {staging}", flush=True)


def process_session(session, options):
    ctx = Context(Path(session))
    return [rectify_episode(ctx, episode, **options) for episode in ctx.episodes]
