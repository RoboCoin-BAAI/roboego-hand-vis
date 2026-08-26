from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "conf" / "datasets"


@dataclass(frozen=True)
class LauncherRequest:
    dataset: str
    session: Path
    stage: str
    config: Path | None = None
    hand_window: str | None = None
    all_hand_windows: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class StageInvocation:
    pipeline: str
    node: str
    selector: str
    params: dict[str, Any]
    kedro_bin: str
    context: dict[str, str]


def load_dataset_config(dataset: str, config_path: Path | None = None) -> dict[str, Any]:
    path = Path(config_path).expanduser() if config_path is not None else DEFAULT_CONFIG_DIR / f"{dataset}.yml"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset config must be a mapping: {path}")
    return payload


def build_stage_invocations(request: LauncherRequest) -> list[StageInvocation]:
    config = load_dataset_config(request.dataset, request.config)
    stages = config.get("stages")
    if not isinstance(stages, dict) or request.stage not in stages:
        available = ", ".join(sorted(stages or {}))
        raise ValueError(f"Unknown stage {request.stage!r}. Available stages: {available}")
    stage_config = stages[request.stage] or {}
    scope = str(stage_config.get("scope") or "session")
    contexts = _build_contexts(config=config, request=request, scope=scope)
    return [_build_invocation(config, stage_config, context) for context in contexts]


def run_stage_invocations(invocations: list[StageInvocation], *, dry_run: bool = False) -> list[str]:
    commands = [render_invocation_command(invocation) for invocation in invocations]
    if dry_run:
        for command in commands:
            print(command)
        return commands
    for invocation in invocations:
        env = os.environ.copy()
        env["KEDRO_DISABLE_TELEMETRY"] = "1"
        env["PYTHONPATH"] = "."
        subprocess.run(_invocation_argv(invocation), check=True, cwd=PROJECT_ROOT, env=env)
    return commands


def render_invocation_command(invocation: StageInvocation) -> str:
    argv = _invocation_argv(invocation)
    return "KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=. " + " ".join(shlex.quote(part) for part in argv)


def _build_invocation(
    dataset_config: dict[str, Any],
    stage_config: dict[str, Any],
    context: dict[str, str],
) -> StageInvocation:
    pipeline = _required_stage_value(stage_config, "pipeline")
    node = _required_stage_value(stage_config, "node")
    selector = str(stage_config.get("selector") or "--nodes")
    namespace = str(stage_config.get("param_namespace") or node)
    runtime = dataset_config.get("runtime") or {}
    kedro_bin = str(runtime.get("kedro_bin") or "kedro")
    params = _render_params(namespace=namespace, params=stage_config.get("params") or {}, context=context)
    extra_params = stage_config.get("extra_params") or {}
    if isinstance(extra_params, dict):
        for extra_namespace, namespace_params in extra_params.items():
            if isinstance(namespace_params, dict):
                params.update(_render_params(namespace=str(extra_namespace), params=namespace_params, context=context))
    return StageInvocation(
        pipeline=pipeline,
        node=node,
        selector=selector,
        params=params,
        kedro_bin=kedro_bin,
        context=context,
    )


def _invocation_argv(invocation: StageInvocation) -> list[str]:
    return [
        invocation.kedro_bin,
        "run",
        "--pipelines",
        invocation.pipeline,
        invocation.selector,
        invocation.node,
        "--params",
        _params_cli_value(invocation.params),
    ]


def _params_cli_value(params: dict[str, Any]) -> str:
    return ",".join(f"{key}={_format_param_value(value)}" for key, value in params.items())


def _format_param_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def _render_params(namespace: str, params: dict[str, Any], context: dict[str, str]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in params.items():
        rendered[f"{namespace}.{key}"] = _render_value(value, context)
    return rendered


def _render_value(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [_render_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, context) for key, item in value.items()}
    return value


def _build_contexts(
    *,
    config: dict[str, Any],
    request: LauncherRequest,
    scope: str,
) -> list[dict[str, str]]:
    session = request.session.expanduser().resolve()
    base_context = _base_context(config, session)
    if scope == "session":
        return [base_context]
    if scope != "hand_window":
        raise ValueError(f"Unsupported stage scope: {scope}")
    hand_windows = _select_hand_windows(session, request.hand_window, request.all_hand_windows)
    return [_hand_window_context(base_context, path) for path in hand_windows]


def _base_context(config: dict[str, Any], session: Path) -> dict[str, str]:
    dataset = config.get("dataset") or {}
    fps = str(dataset.get("fps") or 30)
    paths = config.get("paths") or {}
    context = {
        "session": str(session),
        "session_name": session.name,
        "fps": fps,
        "annotation_dir": str(session / "annotation"),
        "segments_root": str(session / "segments"),
        "num_frames": _infer_num_frames(session, fps=fps),
        "image_height": _infer_image_height(session),
    }
    for key, value in paths.items():
        if isinstance(value, str):
            context[key] = value.format(**context)
    return context


def _hand_window_context(base_context: dict[str, str], hand_window: Path) -> dict[str, str]:
    context = dict(base_context)
    context["hand_window"] = str(hand_window)
    context["hand_window_id"] = hand_window.name
    context["segment"] = str(hand_window.parents[1])
    context["segment_id"] = hand_window.parents[1].name
    context["manifest"] = str(hand_window / "dataset_manifest/dataset_manifest.json")
    context["depth_dir"] = str(hand_window / "depth/fast_foundation")
    context["hamer_run"] = str(hand_window / "hand/hamer_raw")
    context["filter_dir"] = str(hand_window / "hand/filtered")
    context["joints_gate_dir"] = str(hand_window / "hand/joints21_pose_gate")
    context["anchored_dir"] = str(hand_window / "hand/joints21_depth_anchored")
    context["final_dir"] = str(hand_window / "hand/final_hand_result")
    context["num_frames"] = _infer_num_frames(hand_window, fps=base_context["fps"])
    context["image_height"] = _infer_image_height(hand_window)
    return context


def _select_hand_windows(session: Path, hand_window: str | None, all_hand_windows: bool) -> list[Path]:
    if hand_window:
        path = Path(hand_window).expanduser()
        if not path.is_absolute():
            matches = _find_hand_windows(session, name=hand_window)
            if not matches:
                path = session / hand_window
            elif len(matches) == 1:
                path = matches[0]
            else:
                raise ValueError(f"Hand window {hand_window!r} matched multiple paths: {matches}")
        return [path.resolve()]

    matches = _find_hand_windows(session)
    if all_hand_windows:
        if not matches:
            raise FileNotFoundError(f"No hand windows found under {session / 'segments'}")
        return matches
    if len(matches) == 1:
        return matches
    if not matches:
        raise FileNotFoundError(f"No hand windows found under {session / 'segments'}")
    raise ValueError("Found multiple hand windows; pass --hand-window or --all-hand-windows")


def _find_hand_windows(session: Path, name: str | None = None) -> list[Path]:
    root = session / "segments"
    if not root.exists():
        return []
    pattern = f"*/hand_windows/{name}" if name else "*/hand_windows/handwin_*"
    return sorted(path.resolve() for path in root.glob(pattern) if path.is_dir())


def _infer_num_frames(root: Path, *, fps: str) -> str:
    for manifest in (root / "dataset_manifest/dataset_manifest.json", root / "processed/dataset_manifest/dataset_manifest.json"):
        if manifest.is_file():
            try:
                import json

                payload = json.loads(manifest.read_text(encoding="utf-8"))
                hand_window_range = (
                    payload.get("metadata", {})
                    .get("hand_window", {})
                    .get("local_frame_range", {})
                )
                if isinstance(hand_window_range, dict) and hand_window_range.get("num_frames"):
                    return str(int(hand_window_range["num_frames"]))
                frame_count = payload.get("frame_count") or payload.get("num_frames")
                if frame_count:
                    return str(int(frame_count))
                videos = payload.get("videos")
                if isinstance(videos, dict):
                    left_video = videos.get("left")
                    if isinstance(left_video, dict) and left_video.get("frame_count"):
                        return str(int(left_video["frame_count"]))
            except Exception:
                pass
    return str(int(float(fps) * 60))


def _infer_image_height(root: Path) -> str:
    for manifest in (root / "dataset_manifest/dataset_manifest.json", root / "processed/dataset_manifest/dataset_manifest.json"):
        if manifest.is_file():
            try:
                import json

                payload = json.loads(manifest.read_text(encoding="utf-8"))
                for key in ("height", "image_height"):
                    if payload.get(key):
                        return str(int(payload[key]))
                videos = payload.get("videos")
                if isinstance(videos, dict):
                    left_video = videos.get("left")
                    if isinstance(left_video, dict) and left_video.get("height"):
                        return str(int(left_video["height"]))
                cameras = payload.get("cameras")
                if isinstance(cameras, dict):
                    for camera in cameras.values():
                        if isinstance(camera, dict) and camera.get("height"):
                            return str(int(camera["height"]))
            except Exception:
                pass
    return "1050"


def _required_stage_value(stage_config: dict[str, Any], key: str) -> str:
    value = stage_config.get(key)
    if not value:
        raise ValueError(f"Stage config missing required key: {key}")
    return str(value)
