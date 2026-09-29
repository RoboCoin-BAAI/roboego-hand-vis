import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("HAND_DOCKER_TEST") == "1", "requires hand image and NVIDIA Docker runtime")
class LauncherMountTests(unittest.TestCase):
    def test_fresh_checkout_cache_mount(self):
        launcher = Path(__file__).resolve().parents[1] / "run.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ego = root / "ego"
            ego.mkdir()
            (ego / "run.py").touch()
            (ego / "_DATA").mkdir()
            env = dict(os.environ, EGO_ROOT=str(ego), INPUT_DIR=str(root),
                       OUTPUT_DIR=str(root / "output"), TMPDIR=str(root))
            command = """
set -eu
test -d /home/user/EgoHandKit/test_data
touch /home/user/EgoHandKit/test_data/cache_probe
if touch /home/user/EgoHandKit/source_probe 2>/dev/null; then exit 1; fi
"""
            result = subprocess.run(
                ["bash", str(launcher), "bash", "-c", command],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list((ego / "test_data").iterdir()), [])
            self.assertEqual(list(root.glob("roboego-hand-frames-*")), [])


if __name__ == "__main__":
    unittest.main()
