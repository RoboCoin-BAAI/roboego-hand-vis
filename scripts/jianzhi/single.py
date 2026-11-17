"""Run one KS3 Jianzhi LeRobot session through the existing pipeline."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.jianzhi import pipeline
from roboego_hand_vis.upload_data.ks3 import KS3, prefix

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source', required=True, help='ks3://baai-eai-pwm/jianzhi/.../<32-char ID>/ LeRobot directory')
    parser.add_argument('--batch', default='batch_0000000', help='Batch name used in local/state/upload paths')
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--ks3-tool', type=Path, required=True)
    parser.add_argument('--ks3-config', type=Path, default=pipeline.BASE/'utils/ks3/ks3util-config-team')
    parser.add_argument('--upload-ks3-config', type=Path, help='Upload config; defaults to ks3util-config-team beside download config')
    parser.add_argument('--argus-repo', type=Path, default=REPO.parent/'argus')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg') or 'ffmpeg')
    parser.add_argument('--model', default='qwen3.7-plus')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        source = prefix(args.source)
        name = source.rstrip('/').rsplit('/', 1)[-1]
        if (not source.startswith('ks3://baai-eai-pwm/jianzhi/')
                or not re.fullmatch('[0-9a-f]{32}', name)):
            raise ValueError('Source must be a Jianzhi KS3 LeRobot session directory, not a raw MCAP')
        if not re.fullmatch(r'batch_\d+', args.batch):
            raise ValueError('batch must match batch_<digits>')
        if not args.model.startswith(('qwen', 'dashscope/')):
            raise ValueError('Expected a Qwen model')
        if args.upload_ks3_config is None:
            args.upload_ks3_config = args.ks3_config.parent / 'ks3util-config-team'
        for key in ('local_root','state_root','argus_repo','ks3_tool','ks3_config','upload_ks3_config'):
            setattr(args, key, getattr(args, key).expanduser().resolve())
        if args.local_root.is_relative_to(args.state_root) or args.state_root.is_relative_to(args.local_root):
            raise ValueError('Local and state roots must be separate, non-overlapping directories')
        work = args.local_root / args.batch / name
        if args.argus_repo.is_relative_to(args.local_root) or args.local_root.is_relative_to(args.argus_repo):
            raise ValueError('Argus repository must be separate from the local data root')
        for path in (args.ks3_tool, args.ks3_config, args.upload_ks3_config):
            if path.is_relative_to(work):
                raise ValueError('Tool and config must be outside the dataset cleanup directory')
        args.destination = pipeline.DESTINATION
        stages = ['download', 'rectify', 'annotate', 'upload', 'cleanup']
        plan = dict(source=source, local=str(work), destination=prefix(args.destination)+args.batch+'/'+name+'/',
                    state=str(args.state_root/args.batch/'records'/(name+'.json')),
                    logs=str(args.state_root/args.batch/'logs'/name), stages=stages,
                    download_jobs='base default (64)', model=args.model,
                    annotation_mode='existing_text_only', delete_after_verified_upload=True)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            return 0
        for path in (args.ks3_tool, args.ks3_config, args.upload_ks3_config, args.argus_repo/'review/jianzhi/structure_existing.py',
                     args.argus_repo/'review/schemas/episode_structured.schema.json'):
            if not path.is_file():
                raise ValueError('Missing required file: '+str(path))
        if not shutil.which(args.ffmpeg) and not Path(args.ffmpeg).is_file():
            raise ValueError('ffmpeg unavailable; provide --ffmpeg /path/to/ffmpeg')
        client = KS3(args.ks3_tool, args.upload_ks3_config)
        with pipeline.lock(args.state_root):
            for stage in stages:
                print('== '+stage, flush=True)
                record = pipeline.process(source, args.batch, args, client, stage=stage)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if record.get('error'):
                    print('Stopped; local data and state retained. Inspect the stage log and state record.', flush=True)
                    return 1
            return int(record.get('status') != 'complete')
    except Exception as exc:
        print(json.dumps(dict(status='failed', error=str(exc)), ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
