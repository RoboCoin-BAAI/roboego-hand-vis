from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import yaml

from roboego_hand_vis.pipeline_state import read_pipeline_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "conf" / "datasets"


class WorkflowBlockedError(RuntimeError):
    """Raised when a configured gate intentionally blocks later workflow steps."""


@dataclass(frozen=True)
class LauncherRequest:
    dataset: str
    session: Path
    stage: str
    config: Path | None = None
    unit: str | None = None
    hand_window: str | None = None
    all_hand_windows: bool = False
    dry_run: bool = False
    force: bool = False


@dataclass(frozen=True)
class StageInvocation:
    pipeline: str
    node: str
    selector: str
    params: dict[str, Any]
    kedro_bin: str
    context: dict[str, str]


WorkflowStageCallback = Callable[[str, str, list[StageInvocation], BaseException | None], None]


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
    if str(stage_config.get("strategy") or "") == "duration_materialization":
        target_stage, strategy, duration_s = _resolve_duration_materialization_stage(
            stages=stages,
            stage_config=stage_config,
            session=request.session.expanduser().resolve(),
        )
        invocations = build_stage_invocations(replace(request, stage=target_stage))
        return [
            replace(
                invocation,
                context={
                    **invocation.context,
                    "materialization_strategy": strategy,
                    "materialization_target_stage": target_stage,
                    "materialization_duration_s": f"{duration_s:.3f}",
                },
            )
            for invocation in invocations
        ]
    scope = str(stage_config.get("scope") or "session")
    contexts = _build_contexts(config=config, request=request, scope=scope)
    return [_build_invocation(config, stage_config, context) for context in contexts]


def run_launcher_request(request: LauncherRequest, stage_callback: WorkflowStageCallback | None = None) -> list[str]:
    config = load_dataset_config(request.dataset, request.config)
    workflows = config.get("workflows")
    if isinstance(workflows, dict) and request.stage in workflows:
        return _run_workflow_request(
            config=config,
            request=request,
            workflow=workflows[request.stage] or {},
            stage_callback=stage_callback,
        )
    invocations = build_stage_invocations(request)
    return run_stage_invocations(invocations, dry_run=request.dry_run)


def workflow_stage_names(request: LauncherRequest) -> list[str]:
    config = load_dataset_config(request.dataset, request.config)
    workflows = config.get("workflows")
    if not isinstance(workflows, dict) or request.stage not in workflows:
        return [request.stage]
    steps = _workflow_steps(config=config, workflow=workflows[request.stage] or {}, session=request.session.expanduser().resolve())
    names: list[str] = []
    for step in steps:
        for name in _workflow_step_state_stages(step):
            if name not in names:
                names.append(name)
    return names


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


def _run_workflow_request(
    *,
    config: dict[str, Any],
    request: LauncherRequest,
    workflow: dict[str, Any],
    stage_callback: WorkflowStageCallback | None = None,
) -> list[str]:
    commands: list[str] = []
    steps = _workflow_steps(config=config, workflow=workflow, session=request.session.expanduser().resolve())
    state = read_pipeline_state(request.session)
    for step in steps:
        step_stage = str(step["stage"])
        state_stages = _workflow_step_state_stages(step)
        if _workflow_step_already_succeeded(state, state_stages) and not request.force:
            continue
        step_request = replace(
            request,
            stage=step_stage,
            unit=step.get("unit") or request.unit,
            hand_window=step.get("hand_window") or request.hand_window,
            all_hand_windows=bool(step.get("all_hand_windows", False)),
        )
        invocations = build_stage_invocations(step_request)
        if stage_callback and not request.dry_run:
            for state_stage in state_stages:
                stage_callback(state_stage, "processing", invocations, None)
        try:
            step_commands = run_stage_invocations(invocations, dry_run=request.dry_run)
            if not request.dry_run:
                _check_workflow_step_outputs(request=step_request, invocations=invocations)
        except WorkflowBlockedError as exc:
            if stage_callback and not request.dry_run:
                for state_stage in state_stages:
                    stage_callback(state_stage, "blocked", invocations, exc)
            raise
        except BaseException as exc:
            if stage_callback and not request.dry_run:
                for state_stage in state_stages:
                    stage_callback(state_stage, "fail", invocations, exc)
            raise
        commands.extend(step_commands)
        if stage_callback and not request.dry_run:
            for state_stage in state_stages:
                stage_callback(state_stage, "success", invocations, None)
    return commands


def _workflow_step_already_succeeded(state: dict[str, Any], state_stages: list[str]) -> bool:
    stages = state.get("stages", {})
    if not isinstance(stages, dict):
        return False
    return all(isinstance(stages.get(stage), dict) and stages[stage].get("status") == "success" for stage in state_stages)


def _check_workflow_step_outputs(*, request: LauncherRequest, invocations: list[StageInvocation]) -> None:
    if request.stage != "qwen_annotation":
        return
    for invocation in invocations:
        checkpoint_json = _first_param_value(invocation.params, ".checkpoint_json")
        if checkpoint_json and not Path(checkpoint_json).exists():
            raise WorkflowBlockedError(_qwen_checkpoint_missing_message(request.session, Path(checkpoint_json)))


def _first_param_value(params: dict[str, Any], suffix: str) -> str:
    for key, value in params.items():
        if str(key).endswith(suffix) and value not in (None, ""):
            return str(value)
    return ""


def _qwen_checkpoint_missing_message(session: Path, checkpoint_json: Path) -> str:
    session = Path(session).expanduser().resolve()
    lines = [
        "",
        "========== BASE PREPARE BLOCKED ==========",
        "base_prepare blocked: Qwen annotation checkpoint was not produced.",
        f"missing_checkpoint={checkpoint_json}",
    ]
    raw_report = session / "annotation" / "quality_report.json"
    if raw_report.exists():
        try:
            report = json.loads(raw_report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            lines.append(f"raw_quality_report={raw_report} (invalid json)")
        else:
            gate = report.get("raw_headband_quality_gate") if isinstance(report, dict) else {}
            if isinstance(gate, dict) and gate.get("passed") is False:
                lines.append(
                    "raw_headband_quality_gate="
                    f"{gate.get('status', 'failed')} failed_checks={gate.get('failed_checks', [])}"
                )
                for check in gate.get("checks", []):
                    if not isinstance(check, dict) or check.get("passed") is not False:
                        continue
                    detail = f"- {check.get('name')}: {check.get('message', '')}"
                    if "value" in check:
                        detail += f" value={check.get('value')}"
                    if "threshold" in check:
                        detail += f" threshold={check.get('threshold')}"
                    lines.append(detail)
            lines.append(f"raw_quality_report={raw_report}")
    else:
        lines.append(f"raw_quality_report={raw_report} (missing)")
    video_report = session / "annotation" / "video_simple_quality_left.json"
    if video_report.exists():
        try:
            report = json.loads(video_report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            lines.append(f"video_simple_quality_report={video_report} (invalid json)")
        else:
            if isinstance(report, dict) and report.get("passed") is False:
                failed_intervals = report.get("failed_intervals", [])
                lines.append(
                    "video_simple_quality_gate="
                    f"{report.get('status', 'failed')} failed_interval_count={len(failed_intervals)}"
                )
                for interval in failed_intervals:
                    if not isinstance(interval, dict):
                        continue
                    start = float(interval.get("start", 0.0) or 0.0)
                    end = float(interval.get("end", start) or start)
                    detail = f"- {interval.get('label')}: {start:.3f}-{end:.3f}s"
                    if interval.get("reason"):
                        detail += f" | {interval.get('reason')}"
                    lines.append(detail)
            lines.append(f"video_simple_quality_report={video_report}")
    else:
        lines.append(f"video_simple_quality_report={video_report} (missing; likely skipped by raw gate)")
    lines.append("=============================================")
    return "\n".join(lines)


def _workflow_steps(*, config: dict[str, Any], workflow: dict[str, Any], session: Path) -> list[dict[str, Any]]:
    strategy = str(workflow.get("strategy") or "")
    if strategy in {"duration_depth_workflow", "duration_prepare_workflow"}:
        duration_s = _infer_session_duration_s(session)
        short_max_s = float(workflow.get("short_max_s", 180.0))
        medium_max_s = float(workflow.get("medium_max_s", 900.0))
        if duration_s < short_max_s:
            duration_steps = workflow.get("short_steps") or []
        elif duration_s <= medium_max_s:
            duration_steps = workflow.get("medium_steps") or []
        else:
            duration_steps = workflow.get("long_steps") or []
        raw_steps = list(workflow.get("common_steps") or []) + list(duration_steps)
    else:
        raw_steps = workflow.get("steps") or []
    return [_normalize_workflow_step(step) for step in raw_steps]


def _normalize_workflow_step(step: Any) -> dict[str, Any]:
    if isinstance(step, str):
        return {"stage": step}
    if isinstance(step, dict) and step.get("stage"):
        return dict(step)
    raise ValueError(f"Invalid workflow step: {step!r}")


def _workflow_step_state_stages(step: dict[str, Any]) -> list[str]:
    state_stages = step.get("state_stages")
    if isinstance(state_stages, list) and state_stages:
        return [str(stage) for stage in state_stages]
    return [str(step["stage"])]


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
        return "|".join(str(item) for item in value)
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


def _resolve_duration_materialization_stage(
    *,
    stages: dict[str, Any],
    stage_config: dict[str, Any],
    session: Path,
) -> tuple[str, str, float]:
    duration_s = _infer_session_duration_s(session)
    short_max_s = float(stage_config.get("short_max_s", 180.0))
    medium_max_s = float(stage_config.get("medium_max_s", 900.0))
    if duration_s < short_max_s:
        strategy = "session"
        target_stage = str(stage_config.get("short_stage") or "materialize_session")
    elif duration_s <= medium_max_s:
        strategy = "hand_windows"
        target_stage = str(stage_config.get("medium_stage") or "materialize_hand_windows")
    else:
        strategy = "segment_hand_windows"
        target_stage = str(stage_config.get("long_stage") or "materialize_segment_hand_windows")
    if target_stage not in stages:
        raise ValueError(f"Auto materialization target stage {target_stage!r} is not configured")
    return target_stage, strategy, duration_s


def _infer_session_duration_s(session: Path) -> float:
    video_path = Path(session) / "videos" / "left.mp4"
    if video_path.is_file():
        duration_s = _infer_video_duration_s(video_path)
        if duration_s > 0:
            return duration_s

    meta_path = Path(session) / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed meta.json for auto materialization: {meta_path}") from exc
        for key in ("duration_sec", "span_sec"):
            value = meta.get(key)
            if value is not None:
                duration_s = float(value)
                if duration_s > 0:
                    return duration_s

    raise ValueError(
        "Cannot infer session duration for auto materialization. "
        f"Expected positive duration_sec/span_sec in {meta_path} or readable {video_path}."
    )


def _infer_video_duration_s(video_path: Path) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        try:
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                if fps > 0 and frame_count > 0:
                    return frame_count / fps
        finally:
            cap.release()
    except Exception as exc:
        raise ValueError(f"Failed to infer session duration from video: {video_path}") from exc
    return 0.0


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
    if scope == "prepared_unit":
        return _build_prepared_unit_contexts(base_context=base_context, request=request)
    if scope == "unit":
        return _build_unit_contexts(base_context=base_context, request=request)
    if scope != "hand_window":
        raise ValueError(f"Unsupported stage scope: {scope}")
    hand_windows = _select_hand_windows(session, request.hand_window, request.all_hand_windows)
    return [_hand_window_context(base_context, path) for path in hand_windows]


def _build_unit_contexts(*, base_context: dict[str, str], request: LauncherRequest) -> list[dict[str, str]]:
    if request.unit:
        return [_unit_context(base_context, Path(request.unit).expanduser())]
    if request.hand_window or request.all_hand_windows:
        hand_windows = _select_hand_windows(Path(base_context["session"]), request.hand_window, request.all_hand_windows)
        return [_unit_context(base_context, path) for path in hand_windows]
    return [_unit_context(base_context, Path(base_context["session"]))]


def _build_prepared_unit_contexts(*, base_context: dict[str, str], request: LauncherRequest) -> list[dict[str, str]]:
    if request.unit or request.hand_window or request.all_hand_windows:
        return _build_unit_contexts(base_context=base_context, request=request)
    session = Path(base_context["session"])
    units = _find_prepared_units(session)
    if not units:
        raise FileNotFoundError(f"No prepared dataset_manifest.json found under {session}")
    return [_unit_context(base_context, unit) for unit in units]


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
    return _unit_context(base_context, hand_window)


def _unit_context(base_context: dict[str, str], unit: Path) -> dict[str, str]:
    unit = unit.resolve()
    context = dict(base_context)
    context["unit"] = str(unit)
    context["unit_id"] = unit.name
    context["hand_window"] = str(unit)
    context["hand_window_id"] = unit.name
    if unit.parent.name == "hand_windows" and unit.parents[1].name != Path(base_context["session"]).name:
        segment = unit.parents[1]
    else:
        segment = unit
    context["segment"] = str(segment)
    context["segment_id"] = segment.name
    context["manifest"] = str(unit / "dataset_manifest/dataset_manifest.json")
    context["depth_dir"] = str(unit / "depth/fast_foundation")
    context["hamer_run"] = str(unit / "hand/hamer_raw")
    context["filter_dir"] = str(unit / "hand/filtered")
    context["joints_gate_dir"] = str(unit / "hand/joints21_pose_gate")
    context["anchored_dir"] = str(unit / "hand/joints21_depth_anchored")
    context["final_dir"] = str(unit / "hand/final_hand_result")
    context["num_frames"] = _infer_num_frames(unit, fps=base_context["fps"])
    context["image_height"] = _infer_image_height(unit)
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
            raise FileNotFoundError(f"No hand windows found under {session / 'segments'} or {session / 'hand_windows'}")
        return matches
    if len(matches) == 1:
        return matches
    if not matches:
        raise FileNotFoundError(f"No hand windows found under {session / 'segments'} or {session / 'hand_windows'}")
    raise ValueError("Found multiple hand windows; pass --hand-window or --all-hand-windows")


def _find_hand_windows(session: Path, name: str | None = None) -> list[Path]:
    matches: list[Path] = []
    segments_root = session / "segments"
    if segments_root.exists():
        if name:
            patterns = [f"*/hand_windows/{name}", f"*/hand_windows/{name}_*"]
        else:
            patterns = ["*/hand_windows/handwin_*"]
        for pattern in patterns:
            matches.extend(path.resolve() for path in segments_root.glob(pattern) if path.is_dir())
    direct_root = session / "hand_windows"
    if direct_root.exists():
        patterns = [name, f"{name}_*"] if name else ["handwin_*"]
        for pattern in patterns:
            if pattern:
                matches.extend(path.resolve() for path in direct_root.glob(pattern) if path.is_dir())
    return sorted(set(matches))


def _find_prepared_units(session: Path) -> list[Path]:
    search_groups = [
        session / "hand_windows",
        session / "segments",
        session / "segments",
        session,
    ]
    manifest_patterns = [
        "handwin_*/dataset_manifest/dataset_manifest.json",
        "*/hand_windows/handwin_*/dataset_manifest/dataset_manifest.json",
        "seg_*/dataset_manifest/dataset_manifest.json",
        "dataset_manifest/dataset_manifest.json",
    ]
    for root, pattern in zip(search_groups, manifest_patterns):
        if not root.exists():
            continue
        units = sorted({path.parents[1].resolve() for path in root.glob(pattern) if path.is_file()})
        if units:
            return units
    return []


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
