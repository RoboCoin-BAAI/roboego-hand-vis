"""Undistort one converted Archebase dataset, including camera-frame hands."""
import argparse
import json
import shutil
from pathlib import Path

from .compute import process_session


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--balance',type=float,default=0.)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    ffmpeg=args.ffmpeg or shutil.which('ffmpeg')
    if ffmpeg is None and not args.dry_run:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    print(json.dumps(process_session(args.source,balance=args.balance,ffmpeg=ffmpeg,dry_run=args.dry_run),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
