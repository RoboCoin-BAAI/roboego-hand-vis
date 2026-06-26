from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .surface_compensation import orientation_aware_compensate_points


MANO_TO_21 = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
MANO_TIP_VERTEX_IDS = [744, 320, 443, 554, 671]


def install_chumpy_pickle_stubs() -> None:
    if "chumpy" in sys.modules:
        return
    chumpy = types.ModuleType("chumpy")
    chumpy.__path__ = []
    reordering = types.ModuleType("chumpy.reordering")
    ch = types.ModuleType("chumpy.ch")

    class Select:
        pass

    class Ch:
        pass

    reordering.Select = Select
    ch.Ch = Ch
    chumpy.Ch = Ch
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.reordering"] = reordering
    sys.modules["chumpy.ch"] = ch


def load_mano_joint_regressor(model_path: Path) -> np.ndarray:
    install_chumpy_pickle_stubs()
    with model_path.open("rb") as f:
        model = pickle.load(f, encoding="latin1")
    regressor = model["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    regressor = np.asarray(regressor, dtype=np.float32)
    if regressor.shape != (16, 778):
        raise ValueError(f"Unexpected MANO J_regressor shape in {model_path}: {regressor.shape}")
    return regressor


def load_mano_regressors(mano_model_dir: Path) -> Dict[str, np.ndarray]:
    return {
        "ego_left": load_mano_joint_regressor(mano_model_dir / "MANO_LEFT.pkl"),
        "ego_right": load_mano_joint_regressor(mano_model_dir / "MANO_RIGHT.pkl"),
    }


def vertices_to_hamer21_joints(vertices: np.ndarray, regressor_16: np.ndarray) -> np.ndarray:
    if vertices.shape != (778, 3):
        raise ValueError(f"Expected MANO mesh vertices shape (778, 3), got {vertices.shape}")
    mano16 = regressor_16 @ vertices.astype(np.float32)
    tips = vertices[np.asarray(MANO_TIP_VERTEX_IDS, dtype=np.int64)]
    joints21 = np.concatenate([mano16, tips], axis=0)[np.asarray(MANO_TO_21, dtype=np.int64)]
    return joints21.astype(np.float32)


def apply_orientation_compensation(
    vertices_camera: np.ndarray,
    canonical_status: str,
    regressor_by_side: Dict[str, np.ndarray],
    args: Any,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if not args.orientation_aware_compensation:
        return vertices_camera, {"enabled": False}
    regressor = regressor_by_side.get(canonical_status)
    if regressor is None:
        return vertices_camera, {"enabled": False, "reason": f"missing MANO regressor for {canonical_status}"}
    joints21 = vertices_to_hamer21_joints(vertices_camera, regressor)
    compensated, info = orientation_aware_compensate_points(
        vertices_camera,
        joints21,
        depth_bias_m=args.orientation_depth_bias_m,
        surface_min_m=args.orientation_surface_min_m,
        surface_max_m=args.orientation_surface_max_m,
        ray_blend=args.orientation_ray_blend,
    )
    return compensated, {"enabled": True, **info}
