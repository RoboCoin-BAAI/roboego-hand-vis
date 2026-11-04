"""Validate existing QC/rectification artifacts before explicitly adopting them."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def validate(stage, session, level, source=None):
    if stage == 'convert':
        data = json.loads((session / 'meta/l1_source.json').read_text(encoding='utf-8'))
        assert data['source'] == str(Path(source).resolve()), 'Conversion source mismatch'
        assert (session / 'meta/info.json').is_file()
        assert (session / 'meta/episodes.jsonl').stat().st_size > 0
    elif stage == 'download':
        from roboego_hand_vis.download_data.downloader import KS3Downloader
        # Download receipts are intentionally not imported blindly: legacy data
        # has no verifiable receipt. Run download normally or start at QC.
        raise ValueError('Download adoption is unsupported; configure run_stages to start at QC for existing data')
    elif stage == 'points':
        from roboego_hand_vis.lerobot_hand_pose.L3.convert_world_points import reusable
        from roboego_hand_vis.lerobot_hand_pose.L3.world_points import DEFAULT_MAX_GAP_S
        if level != 'L3' or not reusable(session, 'both', DEFAULT_MAX_GAP_S):
            raise ValueError('World hand points failed source/output/extrinsic verification')
    elif stage == 'qc':
        report = json.loads((session / 'annotation/quality_report.json').read_text(encoding='utf-8'))
        assert report['passed'] is True and report['gate']['passed'] is True, 'QC did not pass'
        assert not report['gate']['failed_checks']
        expected = 'lerobot_headband_integrity_l3' if level == 'L3' else 'lerobot_headband_integrity'
        assert report['stage'] == expected, 'QC dataset type mismatch'
        assert report['checks'], 'Empty QC report'
    elif stage == 'rectify':
        import numpy as np
        import yaml
        from roboego_hand_vis.lerobot_headband_integrity.common import Context
        from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import load_geometry
        from roboego_hand_vis.lerobot_stereo_rectification.L1.reuse import validate_existing
        ctx = Context(session)
        for episode in ctx.episodes:
            provenance = None
            if level == 'L3':
                from roboego_hand_vis.lerobot_stereo_rectification.L3.compute import load_geometry as load_l3, CAMERAS
                geometry, provenance = load_l3(session, episode)
                cameras = CAMERAS
            else:
                geometry = load_geometry(session / 'calibrations/camera-camchain.yaml')
                cameras = {'left':'head_left', 'right':'head_right'}
            prefix = f'{episode.parent.name}_{episode.name}_' if len(ctx.episodes)>1 else ''
            directory = session / 'calibrations'
            params_path = directory / f'{prefix}rectify_params.yaml'
            saved = yaml.safe_load(params_path.read_text(encoding='utf-8'))
            params = {k:np.asarray(saved[k]) for k in ('R1','R2','P1','P2','Q')}
            videos = {s:Path('videos')/episode.parent.name/f'observation.images.{cameras[s]}_rectified'/f'{episode.name}.mp4' for s in cameras}
            stamps = {s:episode.relative_to(session)/f'observation.images.{cameras[s]}_rectified.parquet' for s in cameras}
            validate_existing(ctx, episode, manifest_path=directory/f'{prefix}dataset_manifest.json',
                params_path=params_path, mapping_path=directory/f'{prefix}frame_mapping.parquet',
                video_relative=videos, timestamp_relative=stamps, params=params, size=tuple(saved['resolution']),
                crops=saved['input_crops'], output_roi=tuple(saved['crop_roi'][k] for k in ('x_min','y_min','x_max','y_max')),
                transform=geometry[2], balance=saved['balance'], camera_names=cameras, calibration_source=provenance)
    return True


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('qc','rectify','convert','points'))
    p.add_argument('--session',required=True,type=Path)
    p.add_argument('--level',required=True,choices=('L1','L3'))
    p.add_argument('--source',type=Path)
    args=p.parse_args()
    validate(args.stage,args.session.resolve(),args.level,args.source)
