# EGOL1 Windows Quality And Upload Tool

This folder is a standalone Windows tool for SD-card quality checks and uploads.
You can copy this whole `windows_l1_tool` folder to Windows and run the entry file inside it.
It includes the raw headband quality checker locally, including IMU signal checks and duplicate timestamp repair.

## Run

```bat
cd windows_l1_tool
python sd_card_auto_uploader_L1.py
```

Menu:

```text
1. 批量质检
2. 上传
```

Upload menu:

```text
1. 只上传质检通过的
2. 默认逻辑上传
```

## Outputs

- Each session writes `annotation/quality_report.json`.
- Failed quality sessions are summarized in `l1_quality_failed_sessions.json`.
- Sessions skipped by quality-filtered upload are summarized in `raw_headband_quality_skipped_uploads.json`.
- Duplicate timestamp files are repaired automatically when repair limits allow it.

## Dependencies

Install these in the Windows Python environment before running or packaging:

```bat
pip install -r requirements.txt
```

For upload, also install the uploader dependency used in your Windows environment.

## PyInstaller Example

```bat
cd windows_l1_tool
pyinstaller --onefile --name egol1_quality_upload sd_card_auto_uploader_L1.py
```
