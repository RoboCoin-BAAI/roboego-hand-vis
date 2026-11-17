"""Compatibility entry point for the shared KS3 batch downloader."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from roboego_hand_vis.download_data.base.run import main, read_batch

if __name__ == '__main__':
    raise SystemExit(main())
