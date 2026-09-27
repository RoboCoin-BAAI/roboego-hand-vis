"""Stage only required model assets, dereferencing host-specific symlinks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mint-root", type=Path, default=Path.home() / "wuji-ego-mint")
    parser.add_argument("--ego-root", type=Path, default=Path.home() / "EgoHandKit")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / ".build/models")
    args = parser.parse_args()
    files = {
        "mint/model.safetensors": args.mint_root / "checkpoints/model.safetensors",
        "mano/mano_left/MANO_LEFT.pkl": args.mint_root / "assets/mano/mano_left/MANO_LEFT.pkl",
        "mano/mano_right/MANO_RIGHT.pkl": args.mint_root / "assets/mano/mano_right/MANO_RIGHT.pkl",
        "egohandkit/data/mano/MANO_RIGHT.pkl": args.ego_root / "_DATA/data/mano/MANO_RIGHT.pkl",
        "egohandkit/data/mano_mean_params.npz": args.ego_root / "_DATA/data/mano_mean_params.npz",
        "egohandkit/hamer_ckpts/model_config.yaml": args.ego_root / "_DATA/hamer_ckpts/model_config.yaml",
        "egohandkit/hamer_ckpts/checkpoints/hamer.ckpt": args.ego_root / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        parser.error("Missing assets: " + ", ".join(missing))
    records = []
    for relative, source in files.items():
        target = args.output / relative
        source = source.resolve(strict=True)
        if target.resolve() == source:
            raise ValueError(f"Staging output must differ from source: {source}")
        checksum = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or digest(target) != checksum:
            shutil.copyfile(source, target)
        target.chmod(0o644)
        records.append({"path": relative, "bytes": target.stat().st_size, "sha256": checksum})
        print(relative, target.stat().st_size, checksum, flush=True)
    manifest = {
        "schema_version": 1,
        "scope": "MINT inference + canonical HaMeR, no Omega/ViTPose/depth estimator",
        "distribution": "Contains separately licensed MANO and model assets; private use only until redistribution rights are reviewed.",
        "files": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
