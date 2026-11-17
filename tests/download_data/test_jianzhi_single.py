import json
from scripts.jianzhi import single


def arguments(tmp_path):
    return ['--source', 'ks3://baai-eai-pwm/jianzhi/business/'+'a'*32+'/',
            '--local-root',str(tmp_path/'local'),'--state-root',str(tmp_path/'state'),
            '--ks3-tool',str(tmp_path/'ks3'),'--ks3-config',str(tmp_path/'config'),
            '--argus-repo',str(tmp_path/'argus'),'--ffmpeg','ffmpeg']


def setup_files(tmp_path, monkeypatch):
    (tmp_path/'ks3').touch()
    (tmp_path/'config').touch()
    (tmp_path/'ks3util-config-team').touch()
    for name in ['review/jianzhi/structure_existing.py','review/schemas/episode_structured.schema.json']:
        path=tmp_path/'argus'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(single.shutil,'which',lambda name: name)


def test_dry_run_does_not_create_work(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(single.pipeline,'process',lambda *a,**k: (_ for _ in ()).throw(AssertionError()))
    assert single.main(arguments(tmp_path)+['--dry-run']) == 0
    plan=json.loads(capsys.readouterr().out)
    assert plan['annotation_mode']=='existing_text_only'
    assert plan['model']=='qwen3.7-plus'
    assert not (tmp_path/'local').exists()
    assert not (tmp_path/'state').exists()


def test_stages_sequential_and_cleanup_last(tmp_path, monkeypatch):
    setup_files(tmp_path,monkeypatch)
    calls=[]
    def process(*args,stage):
        calls.append(stage)
        return dict(status='complete' if stage=='cleanup' else stage)
    monkeypatch.setattr(single.pipeline,'process',process)
    assert single.main(arguments(tmp_path))==0
    assert calls==['download','rectify','annotate','upload','cleanup']


def test_failure_never_uploads_or_cleans(tmp_path,monkeypatch):
    setup_files(tmp_path,monkeypatch)
    calls=[]
    def process(*args,stage):
        calls.append(stage)
        return dict(status='rectified',error='model failed') if stage=='annotate' else dict(status=stage)
    monkeypatch.setattr(single.pipeline,'process',process)
    assert single.main(arguments(tmp_path))==1
    assert calls==['download','rectify','annotate']


def test_overlapping_roots_rejected(tmp_path):
    assert single.main(arguments(tmp_path)+['--state-root',str(tmp_path/'local/state'),'--dry-run'])==1
