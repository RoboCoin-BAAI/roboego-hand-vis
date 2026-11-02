"""Download one CSV session using the existing level-specific batch downloader."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level',required=True,choices=('L1','L3'))
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--tool',required=True)
    p.add_argument('--config',required=True)
    args=p.parse_args()
    entry=Path(__file__).resolve().parents[2]/'roboego_hand_vis/download_data'/args.level/'run.py'
    with tempfile.TemporaryDirectory() as tmp:
        csv_path=Path(tmp)/'session.csv'
        with csv_path.open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['目标路径']);writer.writerow([args.source])
        code=subprocess.run([sys.executable,str(entry),'--batch-csv',str(csv_path),
            '--ks3-tool',args.tool,'--ks3-config',args.config,'--output-dir',str(args.output)]).returncode
    raise SystemExit(code)
