import sys
from pathlib import Path

import pytest


@pytest.fixture
def ks3_tool(tmp_path):
    executable = tmp_path / "fake_ks3"
    executable.write_text("#!" + sys.executable + "\n" + '''
import json
import sys
import os
import time
from pathlib import Path
args = sys.argv[1:]
mode = Path(args[args.index('-c') + 1]).read_text().strip()
if mode == 'sleep':
    Path(args[args.index('-c') + 1]).with_suffix('.pid').write_text(str(os.getpid()))
    time.sleep(30)
cp = args.index('cp')
source, target = args[cp + 1:cp + 3]
out = Path(target)
out.mkdir(parents=True, exist_ok=True)
if mode == 'error':
    sys.exit(3)
if mode != 'empty':
    (out / 'videos').mkdir()
    (out / 'videos/left.mkv').write_bytes(b'video-data')
reports = Path(args[args.index('--output-dir') + 1])
reports.mkdir(parents=True, exist_ok=True)
if mode == 'partial':
    (reports / 'cp_failed_task.txt').write_text('header\\nfile download failed\\n')
if not source.endswith('/'):
    sys.exit(5)
print(json.dumps({'source': source}))
''')
    executable.chmod(0o755)
    config = tmp_path / "config"
    config.write_text("ok")
    return executable, config
