"""Build a reviewable session manifest without modifying the source dataset."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

DEFAULT_PRIMARY_CAMERA = "observation.images.headband_left_rgb_rectified"


def camera_label(key):
    return key.removeprefix("observation.images.").removeprefix("headband_").removesuffix("_rectified").removesuffix("_rgb")


def compact_feature(feature):
    return {key: value for key, value in feature.items() if key not in {"names", "landmark_order"}}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()] if path.exists() else []


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build(source: Path, context: dict, *, read_collection_metadata=True, use_info_total_frames=False):
    import pyarrow.parquet as pq

    source = source.resolve()
    info = read_json(source / "meta/info.json")
    features = info["features"]
    files = sorted(p for p in source.rglob("*") if p.is_file()
                   and p != source / "meta/readme.json"
                   and not (p.parent == source / "meta" and p.name.startswith(".readme.json.")))
    relative = lambda p: p.relative_to(source).as_posix()
    warnings = []
    annotations = [{"path": relative(p), "data": read_json(p)} for p in files if p.name.endswith("_structured.json")]
    collect = read_lines(source / "meta/collect_info.jsonl") if read_collection_metadata else []
    tasks = read_lines(source / "meta/tasks.jsonl") if read_collection_metadata else []
    streams = {}
    for key, feature in features.items():
        if not key.startswith("observation.") and key != "action":
            continue
        paths = [p for p in files if p.name == key + ".parquet"]
        samples = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
        streams[key] = {
            "feature_key": key, "definition": compact_feature(feature),
            "sample_rate_hz": feature.get("fps", feature.get("sample_rate")),
            "sample_count": samples, "data_paths": [relative(p) for p in paths],
            "status": "present" if samples else ("empty" if paths else "declared_only"),
            "coordinate_frame": feature.get("coordinate_frame"),
            "units": {"translation": feature.get("translation_unit"), "rotation": None, "signal": None},
            "provenance_paths": [],
        }
        side = "left" if key == "observation.hand_left_points" else "right" if key == "observation.hand_right_points" else None
        if side:
            streams[key]["provenance_paths"] = [relative(p) for p in files if p.name == f"hand_{side}_points_provenance.json"]

    cameras = []
    episode_rows = read_lines(source / "meta/episodes.jsonl")
    episode_indices = ([row['episode_index'] for row in episode_rows] if episode_rows
                       else range(info.get('total_episodes', 0)))
    primary = context.get("primary_camera") or DEFAULT_PRIMARY_CAMERA
    for key, feature in features.items():
        if feature.get("dtype") != "video":
            continue
        video_paths = []
        for episode_index in episode_indices:
            path = source / info['video_path'].format(
                episode_chunk=episode_index // info.get('chunks_size', 1000),
                episode_index=episode_index, video_key=key, feature_key=key)
            if path.is_file():
                video_paths.append(relative(path))
        h, w = feature["shape"][:2]
        cameras.append({"id": key, "is_primary": key == primary, "width": w, "height": h,
                        "fps": feature.get("fps", info.get("fps")), "video_paths": video_paths,
                        "status": "present" if video_paths else "declared_only", "parameters": compact_feature(feature)})
    if primary not in {c["id"] for c in cameras}:
        if primary != DEFAULT_PRIMARY_CAMERA:
            raise ValueError(f"Unknown primary_camera: {primary}")
        warnings.append("默认主视角为 headband_left_rgb_rectified，但 info 未声明该流；其分辨率和 FPS 待确认，不沿用原始鱼眼相机参数。")
    groups = {
        "depth": ("深度图或深度点云；需说明深度单位、缩放、对齐相机及算法。", lambda k: "depth" in k),
        "hand_estimation": ("手部关键点或姿态估计，不能仅凭字段名判定裸手来源。", lambda k: "hand_" in k and ("points" in k or "estimate" in k)),
        "head_trajectory": ("头部/头戴跟踪器位置和姿态随时间变化。", lambda k: k.endswith("head_pose")),
        "eye_gaze": ("注视点轨迹；二维 x/y 的单位及参考图像需确认。", lambda k: "gaze" in k),
        "imu": ("加速度和角速度；设备坐标轴、量程及单位需逐设备确认。", lambda k: "imu" in k),
        "hand_trajectory": ("左右手腕跟踪器位姿，不等同于手指骨架。", lambda k: k.endswith("wrist_pose")),
        "emg": ("独立肌电设备信号，不包含 EEG 文件中的辅助 EMG 列。", lambda k: k.endswith("_emg")),
        "eeg": ("脑电信号流及阻抗；信号列和辅助列分开计数。", lambda k: k == "observation.eeg" or "eeg_impedance" in k),
        "glove": ("手套关节角与骨架；具体来源需采集记录或 provenance 佐证。", lambda k: k in ("observation.hand_pose", "observation.hand_skeleton")),
        "bare_hand_estimation": ("明确以裸手视觉估计算法产生的数据，需人工配置特征名。", lambda k: k in context.get("bare_hand_features", [])),
    }
    modalities = {}
    for name, (definition, select) in groups.items():
        selected = [s.copy() for k, s in streams.items() if select(k)]
        modalities[name] = {"definition": definition, "status": "present" if any(s["status"] == "present" for s in selected) else "not_found",
                            "streams": selected}
    for s in modalities["eeg"]["streams"]:
        if s["feature_key"] == "observation.eeg":
            names = features[s["feature_key"]].get("names") or []
            auxiliary = [n for n in names if n.upper() in {"VEOG", "HEOG", "EKG", "ECG", "EMG", "TRIGGER"}]
            s["signal_column_count"] = s["definition"]["shape"][0]
            s["auxiliary_channel_names"] = auxiliary
            s["eeg_channel_count"] = len(names) - len(auxiliary) if names else None
    eeg = next((s for s in modalities["eeg"]["streams"] if s["feature_key"] == "observation.eeg" and s["status"] == "present"), None)
    eeg_count = eeg["eeg_channel_count"] if eeg else 0
    emg_count = sum(s["definition"]["shape"][0] for s in modalities["emg"]["streams"] if s["status"] == "present")
    indexes = [] if use_info_total_frames else [p for p in files if p.name == "master_index.parquet"]
    episodes = []
    for p in indexes:
        t = pq.read_table(p, columns=["timestamp"])["timestamp"].to_pylist()
        t = [v for v in t if v is not None and math.isfinite(v)]
        episodes.append({"index_path": relative(p), "frame_count": pq.ParquetFile(p).metadata.num_rows,
                         "timestamp_span_s": max(t)-min(t) if t else None})
    fps = info.get("fps")
    frames = sum(e["frame_count"] for e in episodes) if episodes else info.get("total_frames")
    duration = frames / fps if frames is not None and fps else None
    if frames != info.get("total_frames"):
        warnings.append("master_index 实际帧数与 info.total_frames 不一致。")
    for a in annotations:
        if duration is not None and any(s.get("end_s", 0) > duration for s in a["data"].get("subtask", [])):
            warnings.append(f"{a['path']} 存在超出名义数据时长的标注，保留原文，需复核。")
    collection_scenes = list(dict.fromkeys(
        r["scene"].strip() for r in collect
        if isinstance(r.get("scene"), str) and r["scene"].strip()
    ))
    scene = {"type": "unknown", "name": collection_scenes[0] if len(collection_scenes) == 1 else None,
             "source": "meta/collect_info.jsonl" if len(collection_scenes) == 1 else None}
    manual_scene = context.get("scene") or {}
    for key in ("type", "name"):
        value = manual_scene.get(key)
        if value is not None and value != "unknown":
            scene[key] = value
            if key == "name":
                scene["source"] = "context.scene.name"
    if len(collection_scenes) > 1:
        warnings.append("collect_info 中存在多个场景名称；请通过 context.scene 指定本条数据的场景。")
    task_name = context.get("task_name") or (tasks[0].get("task") if len(tasks) == 1 else None)
    if task_name:
        task_name = re.sub(r"[\W_]+", "_", task_name).strip("_")
    actions = sorted({v for a in annotations for s in a["data"].get("subtask", []) for v in s.get("action", [])})
    main = next((c for c in cameras if c["is_primary"]), None)
    level = context.get("level") or (source.name.split("-")[0] if re.match(r"L[123]-", source.name) else None)
    parts = [level, scene.get("name") or scene.get("type"), task_name, camera_label(primary),
             f"{main['width']}x{main['height']}" if main else None,
             f"{main['fps']:g}FPS" if main else None,
             f"{context['view_count']}目" if context.get("view_count") is not None else None,
             f"EEG{eeg_count}" if eeg_count is not None else None, f"EMG{emg_count}"]
    display = re.sub(r"[\W_]+", "_", "_".join(str(v) if v is not None else "待确认" for v in parts)).strip("_")
    if not primary:
        warnings.append("主视角未指定；info.master_feature 仅作为同步主流，不作为主视角。")
    return {
        "schema_version": "1.0.0", "dataset_name": display,
        "identity": {"level": level, "original_folder_name": source.name, "source_path": str(source)},
        "task": {"scene": scene, "name": task_name,
                 "description": context.get("task_description") or next((a["data"].get("task_description") for a in annotations), None),
                 "actions": context.get("actions", actions), "objects": context.get("objects", [o for a in annotations for o in a["data"].get("objects", [])]),
                 "annotation_sources": [a["path"] for a in annotations]},
        "statistics": {"duration_s": duration, "duration_method": "master_frame_count / master_fps (nominal)",
                       "frame_count": frames, "master_fps": fps, "episodes": episodes,
                       "size_bytes": sum(p.stat().st_size for p in files), "size_scope": "all regular files under source, including partial/cache files", "file_count": len(files)},
        "cameras": {"count": len(cameras), "primary_camera": primary,
                    "primary_camera_label": camera_label(primary),
                    "primary_camera_status": main["status"] if main else "not_found",
                    "synchronization_master": info.get("master_feature"),
                    "view_count": context.get("view_count"), "items": cameras},
        "channel_counts": {"eeg": eeg_count, "emg": emg_count, "eeg_policy": "exclude named auxiliary channels; not impedance dimensions", "emg_policy": "sum independent EMG streams with nonempty files"},
        "modalities": modalities,
        "coordinate_systems": {"policy": "retain per-stream source definitions; unknown axes/units/transforms remain null",
                               "transform_convention": "T_target_source: p_target = T_target_source @ p_source (homogeneous column vector)",
                               "world_to_camera": None},
        "calibration": {"source": "meta/calibrations.jsonl" if (source / "meta/calibrations.jsonl").is_file() else None},
        "source_metadata": {"tasks": tasks, "collection": collect, "episodes": read_lines(source / "meta/episodes.jsonl")},
        "review": {"warnings": warnings, "pending":
                   (["场景类别（室内/室外）"] if scene["type"] == "unknown" else [])
                   + (["场景名称"] if scene["name"] is None else [])
                   + (["主视角矫正视频的参数与文件"] if main is None else [])
                   + ["几目定义", "各模态未知坐标轴、单位、时间基准", "相机到 OpenVR 世界坐标的外参", "采集物品与实际操作物品的区分"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--context", type=Path, help="Manual semantic fields; see context.template.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error("Output must be outside source so directory size remains reproducible")
    result = build(args.source, read_json(args.context) if args.context else {})
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
