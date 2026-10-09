import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docker/hand-env/prepare_assets.py"


def make_sources(tmp_path):
    mint = tmp_path / "mint"
    ego = tmp_path / "ego"
    paths = [
        mint / "checkpoints/model.safetensors",
        mint / "assets/mano/mano_left/MANO_LEFT.pkl",
        mint / "assets/mano/mano_right/MANO_RIGHT.pkl",
        ego / "_DATA/data/mano/MANO_RIGHT.pkl",
        ego / "_DATA/data/mano_mean_params.npz",
        ego / "_DATA/hamer_ckpts/model_config.yaml",
        ego / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
    ]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode())
    return mint, ego, paths


def test_staging_dereferences_symlinks_and_hashes_assets(tmp_path):
    mint, ego, paths = make_sources(tmp_path)
    original = tmp_path / "external.ckpt"
    original.write_bytes(paths[-1].read_bytes())
    paths[-1].unlink()
    paths[-1].symlink_to(original)
    output = tmp_path / "models"
    command = [sys.executable, str(SCRIPT), "--mint-root", str(mint),
               "--ego-root", str(ego), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["files"]) == 7
    for record in manifest["files"]:
        path = output / record["path"]
        assert not path.is_symlink()
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert paths[-1].is_symlink()
    assert original.read_bytes() == b"fixture-6"
    subprocess.run(command, check=True, capture_output=True)
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_missing_asset_fails_before_staging(tmp_path):
    mint, ego, paths = make_sources(tmp_path)
    paths[0].unlink()
    output = tmp_path / "models"
    result = subprocess.run([sys.executable, str(SCRIPT), "--mint-root", str(mint),
                             "--ego-root", str(ego), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing assets" in result.stderr
    assert not output.exists()
