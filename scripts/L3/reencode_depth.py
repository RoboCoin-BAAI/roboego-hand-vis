"""Re-encode legacy L3 depth MKVs into compressed MP4 and migrate their sidecars."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.guanglun.reencode_depth import main

if __name__ == '__main__':
    sys.exit(main())
