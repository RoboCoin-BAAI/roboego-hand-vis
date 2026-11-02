from scripts.L1 import inventory_gpu_done as inventory


def test_presence_only_all_episodes(tmp_path):
    s=tmp_path/'session'
    for i in range(2):
        ep=f'episode_{i:06d}'
        (s/'data/chunk-000'/ep).mkdir(parents=True)
        out=s/'depth/chunk-000'/ep
        out.mkdir(parents=True)
        (out/'lerobot_depth.json').touch()
    assert inventory.inspect_session(s)['depth_done']
    assert not inventory.inspect_session(s)['hand_done']
    (s/'depth/chunk-000/episode_000001/lerobot_depth.json').unlink()
    assert not inventory.inspect_session(s)['depth_done']
    for i in range(2):
        out=s/'hand/chunk-000'/f'episode_{i:06d}'
        out.mkdir(parents=True)
        (out/'lerobot_hand.json').touch()
        (out/'prediction.npz').touch()
    assert inventory.inspect_session(s)['hand_done']


def test_empty_never_done(tmp_path):
    result=inventory.inspect_session(tmp_path)
    assert not result['depth_done'] and not result['hand_done']
