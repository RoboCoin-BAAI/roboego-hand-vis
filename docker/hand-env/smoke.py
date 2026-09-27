"""Fail-fast GPU/import/asset/renderer probe, not an algorithm quality test."""
import importlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile

import numpy as np
import torch
import torchvision

mode = sys.argv[1]
assert torch.cuda.is_available(), "CUDA unavailable in container"
value = torch.ones((16, 16), device="cuda")
assert torch.isfinite(value @ value).all().item()
boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32, device="cuda")
assert torchvision.ops.nms(boxes, torch.ones(1, device="cuda"), 0.5).numel() == 1

if mode == "mint":
    import decord
    with tempfile.TemporaryDirectory() as directory:
        video = str(Path(directory) / "decode.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=64x64:rate=2", "-frames:v", "2", video], check=True)
        reader = decord.VideoReader(video)
        assert len(reader) == 2 and reader[0].asnumpy().std() > 0
    root = Path("/home/user/wuji-ego-mint")
    sys.path.insert(0, str(root))
    from mint.inference.engine import _ensure_model_train_on_path
    _ensure_model_train_on_path()
    importlib.import_module("models")
    from mint.visualization.mano import _get_mano_model, ensure_mano_weights
    ensure_mano_weights()
    _get_mano_model(True)
    _get_mano_model(False)
else:
    sys.path.insert(0, "/home/user/EgoHandKit")
    importlib.import_module("run")
    import pyrender
    import trimesh
    scene = pyrender.Scene(ambient_light=np.ones(3))
    scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.box()))
    pose = np.eye(4)
    pose[2, 3] = 3
    scene.add(pyrender.PerspectiveCamera(yfov=0.8), pose=pose)
    renderer = pyrender.OffscreenRenderer(64, 64)
    try:
        color, depth = renderer.render(scene)
        assert np.count_nonzero(depth) > 0, "EGL render is blank"
    finally:
        renderer.delete()

print(json.dumps({"environment": mode, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0), "status": "passed"}))
