from pathlib import Path


def test_dataset_pipelines_export_final_rerun():
    export_script = Path("scripts/pipeline/05_export_final_rerun.sh").read_text()
    ropedia_script = Path("scripts/run_ropedia_pipeline.sh").read_text()
    wuxi_script = Path("scripts/run_wuxi_pipeline.sh").read_text()

    assert "scripts/hand/export_final_rerun.py" in export_script
    assert "--tracks_json" in export_script
    assert "--alignment_report" in export_script
    assert "--final_smoothing_window" in export_script
    assert "scripts/pipeline/05_export_final_rerun.sh" in ropedia_script
    assert "scripts/pipeline/05_export_final_rerun.sh" in wuxi_script
