import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from scripts.pipeline import resident


def test_depth_model_loaded_once_across_sessions(tmp_path, monkeypatch):
    from roboego_hand_vis.lerobot_fast_foundation_depth.L1 import engine, compute
    loaded, processed = [], []
    class Engine:
        def __init__(self, options):
            loaded.append(options)
        def process(self, task):
            processed.append(task)
            if task['session'].endswith('bad'):
                raise ValueError('bad input')
            return dict(frames=10)
    monkeypatch.setattr(engine, 'DepthEngine', Engine)
    monkeypatch.setattr(compute, 'plan_session', lambda source, **kw: [dict(episode=str(source/'episode'))])
    requests=[]
    for name in ('first','bad','last'):
        requests.append(dict(command=['python','run.py','--source',str(tmp_path/name),
                        '--model-root',str(tmp_path/'model'),'--depth-format','hevc12'],
                        log=str(tmp_path/(name+'.log'))))
    monkeypatch.setattr(sys,'argv',['resident.py','depth','L1','0'])
    monkeypatch.setattr(sys,'stdin',io.StringIO(''.join(json.dumps(r)+'\n' for r in requests)))
    output=io.StringIO()
    monkeypatch.setattr(sys,'stdout',output)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    assert resident.main()==0
    assert len(loaded)==1 and len(processed)==3
    assert [json.loads(line)['returncode'] for line in output.getvalue().splitlines()]==[0,1,0]
    assert 'model_load_s=' in (tmp_path/'first.log').read_text()


def test_hand_predictor_loaded_once(tmp_path, monkeypatch):
    import types
    from roboego_hand_vis.lerobot_hand_pose import compute
    loaded, seen=[],[]
    class Predictor:
        def __init__(self, **kwargs):
            loaded.append(kwargs)
            self.config={}
        def run(self,*args,**kwargs): pass
    runtime=types.ModuleType('runtime')
    module=types.ModuleType('runtime.pipeline')
    module.Predictor=Predictor
    monkeypatch.setitem(sys.modules,'runtime',runtime)
    monkeypatch.setitem(sys.modules,'runtime.pipeline',module)
    monkeypatch.setattr(compute,'plans',lambda source,level: [source])
    def process(source,level,**kwargs):
        seen.append(source)
        assert kwargs['inference'] is not None
        kwargs['inference'](source/'video.mp4',source,tmp_path/(source.name+'-infer.log'))
        return []
    monkeypatch.setattr(compute,'process',process)
    requests=[dict(command=['python','run.py','--source',str(tmp_path/name),
                   '--model-root',str(tmp_path)],log=str(tmp_path/(name+'.log'))) for name in ('one','two')]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setattr(sys,'argv',['resident.py','hand','L1','0'])
    monkeypatch.setattr(sys,'stdin',io.StringIO(''.join(json.dumps(r)+'\n' for r in requests)))
    monkeypatch.setattr(sys,'stdout',io.StringIO())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    assert resident.main()==0
    assert len(loaded)==1 and len(seen)==2
