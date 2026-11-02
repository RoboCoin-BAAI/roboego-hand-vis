"""One resident Predictor per GPU process for a preselected episode list."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roboego_hand_vis.lerobot_hand_pose.compute import process
from roboego_hand_vis.lerobot_hand_pose.run import Tee


def run_jobs(request, predictor):
    options = request['options']
    failed = 0
    for number, raw in enumerate(request['tasks'], 1):
        plan = {k:Path(v) if k != 'feature' else v for k,v in raw.items()}
        print(f'[resident-task] {number}/{len(request["tasks"])} {plan["session"].name}/{plan["episode"].name}', flush=True)
        def infer(video, output, log_path):
            with log_path.open('w', encoding='utf-8') as log:
                tee = Tee(sys.stdout, log)
                with redirect_stdout(tee), redirect_stderr(tee):
                    try:
                        predictor.run(video, output, resize_mode=options['resize_mode'],
                                      max_frames=0, render_overlay=False,
                                      progress=lambda value,text:print(f'[model {value:.0%}] {text}',flush=True))
                    except BaseException:
                        traceback.print_exc()
                        raise
            return SimpleNamespace(returncode=0)
        try:
            result = process(plan['session'], request['level'], selected_plans=[plan], inference=infer, **options)
            record = dict(session=str(plan['session']), episode=str(plan['episode']), status='done', results=result)
        except Exception as exc:
            failed += 1
            record = dict(session=str(plan['session']), episode=str(plan['episode']), status='failed', error=str(exc))
        print(json.dumps(record), flush=True)
        with Path(request['results']).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
    return int(bool(failed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    args=parser.parse_args()
    request=json.loads(args.request.read_text(encoding='utf-8'))
    options=request['options']
    sys.path.insert(0,str(Path(options['model_root']).resolve()))
    from runtime.pipeline import Predictor
    print(f'[resident-load] GPU={options["gpu"]}; tasks={len(request["tasks"])}; creating one Predictor',flush=True)
    predictor=Predictor(device='cuda:0',precision=options['precision'])
    predictor.config['window_batch_size']=options['window_batch_size']
    return run_jobs(request,predictor)


if __name__=='__main__':
    raise SystemExit(main())
