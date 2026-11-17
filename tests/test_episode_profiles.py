import math

import numpy as np

from roboego_hand_vis.annotation_statistics.plot_episode_profiles import (
    build_profiles, duration_bin, summarize, subtask_bin)


def test_duration_and_segment_boundaries():
    assert [duration_bin(x) for x in [0,9.9,10,10.1,20,299,300,300.1,600]] == [0,0,0,1,1,29,29,30,30]
    assert [subtask_bin(x) for x in [0,1,2,29,30,31,50]] == [0,1,2,29,30,31,31]


def test_count_shares_and_duration_weighted_mixture():
    profiles = build_profiles([[100,2],[200,3]], [[50,1]])
    assert profiles['L2']['count'] == 3+100/300
    assert math.isclose(profiles['L2']['hours'], profiles['L3']['hours']*4)
    for stats in profiles.values():
        assert math.isclose(sum(stats['duration']), stats['count'])
        assert math.isclose(sum(stats['subtask']), stats['count'])
        assert math.isclose(sum(stats['exact'].values()), stats['count'])


def test_target_hours_and_zero_subtask_exclusion():
    profiles = build_profiles([[15,0],[100,3],[200,4]], [[50,1],[80,2]], ideal=True)
    for level, hours in [('L1',10000),('L2',2000),('L3',500)]:
        assert math.isclose(profiles[level]['hours'], hours)
        assert profiles[level]['exact'].get(0, 0) == 0


def test_doubling_only_changes_guanglun_segment_counts():
    first = [[15,2],[100,3],[200,4]]
    third = [[50,5],[80,6]]
    for ideal in [False, True]:
        original = build_profiles(first, third, ideal=ideal)
        adjusted = build_profiles(first, third, ideal=ideal, guanglun_subtask_factor=2)
        for level in ['L1','L2','L3']:
            assert np.allclose(original[level]['duration'], adjusted[level]['duration'])
            assert original[level]['count'] == adjusted[level]['count']
            assert original[level]['hours'] == adjusted[level]['hours']
        assert adjusted['L3']['exact'] == original['L3']['exact']
        assert adjusted['L1']['exact'] == {2*k:v for k,v in original['L1']['exact'].items()}
    assert first == [[15,2],[100,3],[200,4]]


def test_csv_contains_only_plotted_bars_and_matching_values(tmp_path):
    import csv
    from roboego_hand_vis.annotation_statistics.plot_episode_profiles import export_chart_values
    actual = build_profiles([[100,2],[200,3]], [[50,1]], guanglun_subtask_factor=2)
    ideal = build_profiles([[100,2],[200,3]], [[50,1]], ideal=True, guanglun_subtask_factor=2)
    path = tmp_path/'values.csv'
    export_chart_values(path, actual, ideal)
    with path.with_name('values_actual.csv').open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['panel','bin','level','count','percentage']
        rows = list(reader)
    assert len(rows) == 3*(31+32)
    assert path.with_name('values_ideal.csv').is_file()
    assert not path.exists()
    row = next(row for row in rows if row['panel']=='subtask' and row['level']=='L1' and row['bin']=='4')
    assert float(row['count']) == 1
    assert float(row['percentage']) == 50
