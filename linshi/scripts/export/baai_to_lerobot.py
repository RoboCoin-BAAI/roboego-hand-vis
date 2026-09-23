#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = base_dir / out
    return out.resolve()


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read video frame {frame_idx}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def flat_names(prefix: str) -> list[str]:
    joint_names = [
        "wrist",
        "thumb_1",
        "thumb_2",
        "thumb_3",
        "thumb_tip",
        "index_1",
        "index_2",
        "index_3",
        "index_tip",
        "middle_1",
        "middle_2",
        "middle_3",
        "middle_tip",
        "ring_1",
        "ring_2",
        "ring_3",
        "ring_tip",
        "little_1",
        "little_2",
        "little_3",
        "little_tip",
    ]
    axes = ["x", "y", "z"]
    return [f"{prefix}.{joint}.{axis}" for joint in joint_names for axis in axes]


def fingertip_names(prefix: str) -> list[str]:
    tips = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
    axes = ["x", "y", "z"]
    return [f"{prefix}.{tip}.{axis}" for tip in tips for axis in axes]


def fingertips_from_joints(joints21: np.ndarray) -> np.ndarray:
    return joints21[[4, 8, 12, 16, 20]].reshape(-1).astype(np.float32)


def build_camera_parameters_annotation(manifest: dict, manifest_path: Path) -> dict:
    return {
        "schema_version": "1.0",
        "source_manifest": str(manifest_path),
        "dataset_name": manifest.get("dataset_name"),
        "sequence_name": manifest.get("sequence_name"),
        "reference_camera": manifest.get("reference_camera"),
        "coordinate_frame": "camera_opencv",
        "unit": "meter",
        "videos": manifest.get("videos", {}),
        "camera_model": manifest.get("camera_model", {}),
        "intrinsics": manifest.get("intrinsics", {}),
        "extrinsics": manifest.get("extrinsics", {}),
        "timestamps": manifest.get("timestamps", {}),
        "trajectory": manifest.get("trajectory", {}),
        "notes": {
            "intrinsics": "Pinhole K parameters are expressed in pixels for each exported image/video stream.",
            "extrinsics": "Stereo extrinsics follow the manifest convention; for BAAI this is rectified left camera to right camera.",
            "hand_outputs": "Exported 21-joint hand states are in the reference camera OpenCV frame unless a later world export is used.",
        },
    }


def write_camera_parameters_annotation(manifest: dict, manifest_path: Path, output_root: Path) -> Path:
    annotation_dir = output_root / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = annotation_dir / "camera_parameters.json"
    annotation = build_camera_parameters_annotation(manifest, manifest_path)
    annotation_path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")
    return annotation_path


def parse_quality_report_specs(specs: list[str] | None, base_dir: Path) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Quality report must use name=path format, got: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Quality report name is empty in: {spec}")
        report_path = resolve_path(raw_path.strip(), base_dir)
        if not report_path.exists():
            raise FileNotFoundError(f"Quality report does not exist: {report_path}")
        parsed.append((name, report_path))
    return parsed


def write_quality_annotations(quality_reports: list[tuple[str, Path]], output_root: Path) -> list[Path]:
    if not quality_reports:
        return []

    annotation_dir = output_root / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    used_names: set[str] = set()
    for name, source_path in quality_reports:
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
        if not safe_name:
            safe_name = "quality"
        target_name = source_path.name if source_path.stem.startswith(safe_name) else f"{safe_name}_{source_path.name}"
        suffix = 1
        while target_name in used_names:
            target_name = f"{safe_name}_{suffix}_{source_path.name}"
            suffix += 1
        used_names.add(target_name)

        target_path = annotation_dir / target_name
        shutil.copy2(source_path, target_path)
        written.append(target_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BAAI ego hand result as a local LeRobot v2.1 dataset.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--joints_npz", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--repo_id", default="local/baai-ego-hand")
    parser.add_argument("--task", default="Estimate ego left/right 21 hand joints from rectified stereo video.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=900)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include_right_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_end_effectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--quality_report",
        action="append",
        default=[],
        help="Copy a quality report into annotation using name=path, e.g. stereo=tmp/baai_stereo_quality/stereo_quality_report.json. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    cwd = Path.cwd()
    manifest_path = resolve_path(args.manifest, cwd)
    joints_npz = resolve_path(args.joints_npz, cwd)
    output_root = resolve_path(args.output_root, cwd)
    quality_reports = parse_quality_report_specs(args.quality_report, cwd)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    left_video = Path(manifest["videos"]["left"]["path"]).expanduser().resolve()
    right_video = Path(manifest["videos"]["right"]["path"]).expanduser().resolve()
    width = int(manifest["videos"]["left"]["width"])
    height = int(manifest["videos"]["left"]["height"])
    fps = int(args.fps or round(float(manifest["videos"]["left"]["fps"])))

    joints = np.load(joints_npz)
    left_joints = joints["left_joints_21_camera_opencv"].astype(np.float32)
    right_joints = joints["right_joints_21_camera_opencv"].astype(np.float32)
    left_valid = joints["left_valid"].astype(bool)
    right_valid = joints["right_valid"].astype(bool)
    total_frames = min(args.num_frames, len(left_joints), len(right_joints))

    state_names = flat_names("left") + flat_names("right")
    left_tip_names = fingertip_names("left")
    right_tip_names = fingertip_names("right")
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (126,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (126,),
            "names": state_names,
        },
        "observation.left_hand_joints_valid": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["valid"],
        },
        "observation.right_hand_joints_valid": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["valid"],
        },
        "observation.images.left": {
            "dtype": "video" if args.use_videos else "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
    }
    if args.include_end_effectors:
        features.update(
            {
                "observation.left_hand_end_effectors": {
                    "dtype": "float32",
                    "shape": (15,),
                    "names": left_tip_names,
                },
                "observation.right_hand_end_effectors": {
                    "dtype": "float32",
                    "shape": (15,),
                    "names": right_tip_names,
                },
                "observation.end_effectors": {
                    "dtype": "float32",
                    "shape": (30,),
                    "names": left_tip_names + right_tip_names,
                },
            }
        )
    if args.include_right_video:
        features["observation.images.right"] = {
            "dtype": "video" if args.use_videos else "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=output_root,
        robot_type="ego_hand",
        use_videos=args.use_videos,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    cap_left = cv2.VideoCapture(str(left_video))
    cap_right = cv2.VideoCapture(str(right_video)) if args.include_right_video else None
    try:
        for offset in range(total_frames):
            frame_idx = args.start + offset
            state = np.concatenate([left_joints[offset].reshape(-1), right_joints[offset].reshape(-1)]).astype(np.float32)
            frame = {
                "observation.state": state,
                # For this estimated-pose dataset, action is set to the next pose when available, else current pose.
                "action": (
                    np.concatenate(
                        [
                            left_joints[min(offset + 1, total_frames - 1)].reshape(-1),
                            right_joints[min(offset + 1, total_frames - 1)].reshape(-1),
                        ]
                    ).astype(np.float32)
                ),
                "observation.left_hand_joints_valid": np.asarray([left_valid[offset]], dtype=bool),
                "observation.right_hand_joints_valid": np.asarray([right_valid[offset]], dtype=bool),
                "observation.images.left": read_frame(cap_left, frame_idx),
            }
            if args.include_end_effectors:
                left_tips = fingertips_from_joints(left_joints[offset])
                right_tips = fingertips_from_joints(right_joints[offset])
                frame["observation.left_hand_end_effectors"] = left_tips
                frame["observation.right_hand_end_effectors"] = right_tips
                frame["observation.end_effectors"] = np.concatenate([left_tips, right_tips]).astype(np.float32)
            if cap_right is not None:
                frame["observation.images.right"] = read_frame(cap_right, frame_idx)
            dataset.add_frame(frame, task=args.task, timestamp=float(offset) / float(fps))
        dataset.save_episode()
    finally:
        cap_left.release()
        if cap_right is not None:
            cap_right.release()

    camera_annotation_path = write_camera_parameters_annotation(manifest, manifest_path, output_root)
    quality_annotation_paths = write_quality_annotations(quality_reports, output_root)
    extra = {
        "source_manifest": str(manifest_path),
        "source_joints_npz": str(joints_npz),
        "source_left_video": str(left_video),
        "source_right_video": str(right_video),
        "camera_parameters_annotation": str(camera_annotation_path),
        "quality_report_annotations": [str(path) for path in quality_annotation_paths],
        "coordinate_frame": "camera_opencv",
        "unit": "meter",
        "num_frames": int(total_frames),
        "fps": int(fps),
        "state_layout": "left_joints21_xyz_flat_then_right_joints21_xyz_flat",
        "end_effector_layout": "thumb/index/middle/ring/little fingertips xyz; left then right",
        "action_layout": "next_frame_state; last frame repeats current state",
        "mano_params_note": (
            "Existing BAAI 900 export was generated before HaMeR mano_params were stored. "
            "Re-run HaMeR ego filter after this code change to include full MANO params."
        ),
    }
    (output_root / "meta" / "roboego_source.json").write_text(json.dumps(extra, indent=2, ensure_ascii=False), encoding="utf-8")

    reloaded = LeRobotDataset(args.repo_id, root=output_root, episodes=[0], download_videos=False)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "repo_id": args.repo_id,
                "num_frames": int(reloaded.num_frames),
                "num_episodes": int(reloaded.num_episodes),
                "features": reloaded.features,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
