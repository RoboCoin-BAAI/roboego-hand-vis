"""Manifest-only L1 recovery: QC all, rectify missing, VIO all, review missing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def load_sessions(manifest, root):
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty inventory JSON array')
    sessions = [Path(row['session']).resolve() for row in rows]
    if len(set(sessions)) != len(sessions):
        raise ValueError('Duplicate sessions')
    for s in sessions:
        if (not s.is_relative_to(root) or len(s.relative_to(root).parts) != 2
                or not s.parent.name.startswith('batch_') or not s.name.startswith('cp_')):
            raise ValueError('Invalid session: ' + str(s))
    return sessions


def rectified_present(session):
    """Presence-only check: both rectified videos for every data episode."""
    episodes = [p for p in (session/'data').glob('chunk-*/episode_*') if p.is_dir()]
    return bool(episodes) and all(
        (session/'videos'/episode.parent.name/
         f'observation.images.head_{side}_rectified'/f'{episode.name}.mp4').is_file()
        for episode in episodes for side in ('left','right'))


def semantic_complete(session):
    # A completed primary annotation is retained even if secondary extraction is missing.
    job = session / 'annotation/argus'
    try:
        status = json.loads((job/'run_final/run.json').read_text(encoding='utf-8'))
        outputs = list((job/'run_final/out').glob('episode_*.json'))
        return (status.get('status') == 'done' and bool(outputs)
                and all(isinstance(json.loads(p.read_text(encoding='utf-8')), dict)
                        and bool(json.loads(p.read_text(encoding='utf-8'))) for p in outputs))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('/share/project/liuyou/humancoin/data/L1_existing_paths.json'))
    p.add_argument('--root', type=Path, default=Path('/share/project/liuyou/humancoin/data/L1_lerobot'))
    p.add_argument('--code-root', type=Path, default=REPO.parent)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--review-workers', type=int, default=4)
    p.add_argument('--output', type=Path)
    p.add_argument('--execute', action='store_true', help='Without this flag, inspect manifest counts only')
    a = p.parse_args(argv)
    if min(a.workers, a.review_workers) < 1:
        p.error('Worker counts must be positive')
    root = a.root.resolve()
    sessions = load_sessions(a.manifest, root)
    print(json.dumps(dict(sessions=len(sessions), mode='execute' if a.execute else 'preview',
                         phases=['qc_all','rectify_missing','vio_all','review_missing'])), flush=True)
    if not a.execute:
        return 0
    from scripts.pipeline.check import validate
    from scripts.pipeline.run import Runner, claim, read, write, update_rectified_index, validate_config
    import yaml
    argus = a.code_root / 'argus'
    sys.path.insert(0, str(argus))
    from review.capture_context import dataset_note
    from review.capture_context import snapshot_for_video
    # Per-session locks coordinate with general CPU/GPU scheduler lanes.
    output = a.output or root.parent / 'pipeline_tasks/L1/temporary_runs' / uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    save(output/'sessions.json', list(map(str, sessions)))
    print('Run directory: ' + str(output), flush=True)
    config = yaml.safe_load((REPO/'scripts/L1/pipeline.yaml').read_text(encoding='utf-8'))
    config['paths']['output_root'] = str(root)
    config['paths']['state_root'] = str(root.parent/'pipeline_tasks')
    config['start_at'] = 'qc'
    config['force_stages'] = []
    validate_config(config)
    runner = Runner(config, 'cpu')
    failures = []

    def check(stage, session):
        if stage == 'rectify':
            return rectified_present(session)
        try:
            validate(stage, session, 'L1')
            return True
        except Exception:
            return False

    def command(stage, session, extra=()):
        if stage == 'review':
            videos = sorted((session/'videos').glob('chunk-*/observation.images.head_left_rectified/episode_*.mp4'))
            if len(videos) != 1:
                raise ValueError('Expected exactly one left rectified video: ' + str(session))
            cmd = [sys.executable, '-m', 'review', '--data', str(videos[0]), '--rig', 'ego_head',
                   '--jobs','1','--no-board','--model','qwen3.6-plus','--reasoning','none',
                   '--sample-fps','2','--max-tokens','16384','--concurrency','1','--cap','0']
            context = snapshot_for_video(videos[0])
            note = dataset_note(None, context)
            if note:
                cmd += ['--dataset-note', note]
        else:
            module = {'qc':'lerobot_headband_integrity','rectify':'lerobot_stereo_rectification','vio':'lerobot_vio'}[stage]
            cmd = [sys.executable, str(REPO/'roboego_hand_vis'/module/'L1/run.py'),
                   '--source',str(session),'--workers','1']
            if stage == 'qc':
                cmd += ['--ffprobe',str(Path(sys.executable).parent/'ffprobe'),
                        '--repair-calibration-root',str(root),
                        '--calibration-donor-cache',str(output/'donors.json')]
            elif stage == 'rectify':
                cmd += ['--no-skip-existing','--ffmpeg',str(Path(sys.executable).parent/'ffmpeg')]
            else:
                cmd += ['--openvins-root',str(a.code_root/'open_vins'),'--init-mode','static-then-dynamic']
        return cmd + list(extra)

    def phase(stage, selected, workers):
        def one(session):
            task = dict(batch=session.parent.name,name=session.name,source=str(session))
            try:
                with claim(runner.folder(task)/'session.lock') as acquired:
                    if not acquired:
                        raise RuntimeError('Session is in use by another CPU/GPU runner')
                    if stage == 'qc':
                        states = read(runner.folder(task)/'state.json')
                        runner.invalidate('qc', states)
                        states['qc'] = dict(status='failed', error='Temporary full QC pending completion')
                        write(runner.folder(task)/'state.json', states)
                        update_rectified_index(config, session, False)
                    if stage == 'rectify' and check('rectify', session):
                        status = 'skipped_complete'
                    elif stage == 'review' and semantic_complete(session):
                        return dict(session=str(session),status='skipped_complete')
                    else:
                        cmd = command(stage, session)
                        log = output/stage/session.parent.name/(session.name+'.log')
                        log.parent.mkdir(parents=True,exist_ok=True)
                        save(log.with_suffix('.command.json'), cmd)
                        with log.open('w',encoding='utf-8') as stream:
                            rc = subprocess.run(cmd,cwd=argus if stage=='review' else REPO,
                                                stdout=stream,stderr=subprocess.STDOUT).returncode
                        if rc != 0:
                            raise RuntimeError(f'Exit {rc}; log: {log}')
                        status = 'done'
                    if stage in ('qc','rectify') and not check(stage, session):
                        raise RuntimeError('Output validation failed')
                    if stage in ('qc','rectify'):
                        states = read(runner.folder(task)/'state.json')
                        runner.invalidate(stage, states)
                        states[stage] = dict(status='done',generation=uuid4().hex,task=task,
                            signature=runner.signature(stage,task,states),artifacts=runner.artifacts(stage,task),
                            origin='temporary_rerun',request=runner.run_id)
                        write(runner.folder(task)/'state.json',states)
                        update_rectified_index(config,session,stage=='rectify')
                    return dict(session=str(session),status=status)
            except Exception as e:
                return dict(session=str(session),status='failed',error=str(e))
        print(json.dumps(dict(phase=stage,sessions=len(selected))),flush=True)
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(one,selected):
                results.append(result)
                print(json.dumps(dict(phase=stage,**result)),flush=True)
        save(output/(stage+'.json'),results)
        failures.extend(dict(phase=stage,**r) for r in results if r['status']=='failed')
        save(output/'failures.json', failures)
        return [Path(r['session']) for r in results if r['status']!='failed']

    # All QC completes before rectification; all rectification before VIO; all VIO before review.
    qc_ok = phase('qc', sessions, a.workers)
    rectified = phase('rectify', qc_ok, a.workers)
    phase('vio', rectified, a.workers)  # Always execute, including sessions with existing VIO.
    phase('review', rectified, a.review_workers)  # A VIO failure does not suppress semantic review.
    save(output/'summary.json',dict(total=len(sessions),qc_passed=len(qc_ok),rectified=len(rectified),failures=len(failures)))
    print('Reports: ' + str(output),flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
