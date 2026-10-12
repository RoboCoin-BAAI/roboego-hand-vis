from argparse import Namespace

from roboego_hand_vis.stereo_depth_gate import cli
from roboego_hand_vis.stereo_depth_gate.compute import SummaryQualityConfig, summarize


def config(data, tmp_path):
    return cli.config_from_yaml_and_args(data, Namespace(session=str(tmp_path), manifest='', output_dir='', sample_count=None, min_orb_matches=None))


def row():
    return dict(left_gray_mean=100, right_gray_mean=100, left_sharp_laplacian_var=100,
                right_sharp_laplacian_var=100, left_dark_ratio=0, right_dark_ratio=0,
                left_bright_ratio=0, right_bright_ratio=0, brightness_gap=0,
                orb_matches=50, median_abs_dy=4, p90_abs_dy=20)


def test_defaults_keep_existing_high_risk_release():
    report = summarize([row()], SummaryQualityConfig())
    assert not report['passed']
    assert report['usable_for_depth']
    assert report['depth_risk_level'] == 'high_risk'
    assert all(report['depth_usable_checks'].values())


def test_yaml_relaxed_thresholds_actually_control_gate(tmp_path):
    cfg = config({'depth_usable_thresholds': {'max_median_abs_dy': 3, 'max_p90_abs_dy': 10}}, tmp_path)
    report = summarize([row()], cfg.summary)
    assert not report['usable_for_depth']
    assert report['depth_risk_level'] == 'blocked'
    assert report['usable_for_depth_thresholds'] == dict(max_median_abs_dy=3, max_p90_abs_dy=10)


def test_report_lists_both_sets_and_explains_high_risk(tmp_path, monkeypatch):
    cfg = config({}, tmp_path)
    monkeypatch.setattr(cli, 'load_manifest_video_paths', lambda _: (tmp_path/'left', tmp_path/'right', {}))
    monkeypatch.setattr(cli, 'inspect_video', lambda _: {'frame_count': 1, 'fps': 30})
    monkeypatch.setattr(cli, 'read_frame', lambda *args: None)
    monkeypatch.setattr(cli, 'build_sample_row', lambda *args: row())
    report = cli.run_stereo_depth_gate(cfg)
    assert report['strict_thresholds']['max_median_abs_dy'] == 2
    assert report['depth_usable_thresholds']['max_median_abs_dy'] == 8
    assert report['strict_thresholds']['max_p90_abs_dy'] == 5
    assert report['depth_usable_thresholds']['max_p90_abs_dy'] == 70
    for name in ('min_orb_matches', 'min_sharpness', 'max_brightness_gap', 'max_dark_ratio', 'max_bright_ratio'):
        assert report['strict_thresholds'][name] == report['depth_usable_thresholds'][name]
    assert report['thresholds'] == report['strict_thresholds']
    assert report['gate_decision']['passed'] is True
    assert report['gate_decision']['based_on'] == 'usable_for_depth'
    assert '严格' in report['gate_decision']['explanation']
    assert '风险' in report['gate_decision']['explanation']
    assert (cfg.output_dir/'stereo_depth_gate_report.json').exists()


def test_blocked_risk_when_configured_depth_limits_are_tighter(tmp_path):
    cfg = config({'depth_usable_thresholds': {'max_median_abs_dy': 1, 'max_p90_abs_dy': 3}}, tmp_path)
    sample = dict(row(), median_abs_dy=1.5, p90_abs_dy=4)
    report = summarize([sample], cfg.summary)
    assert report['passed'] is True
    assert report['usable_for_depth'] is False
    assert report['depth_risk_level'] == 'blocked'


def test_legacy_report_explanation_uses_effective_depth_limits(tmp_path):
    from roboego_hand_vis.stereo_depth_gate.compute import next_step_gate_zh
    cfg = config({'depth_usable_thresholds': {'max_median_abs_dy': 1, 'max_p90_abs_dy': 3}}, tmp_path)
    report = summarize([dict(row(), median_abs_dy=1.5, p90_abs_dy=4)], cfg.summary)
    explanation = next_step_gate_zh(report)
    assert '<= 1.0' in explanation['宽松深度可用标准']
    assert '<= 3.0' in explanation['宽松深度可用标准']
    assert '不建议' in explanation['当前建议']
