"""Export native intermediate files to report.json and trajectory.npz."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roboego_hand_vis.vio.delivery import export_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export_native(args.work, args.output)


if __name__ == '__main__':
    main()
