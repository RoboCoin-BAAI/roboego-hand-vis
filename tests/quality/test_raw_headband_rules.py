from roboego_hand_vis.quality.raw_headband_rules import evaluate_quality_report


def test_rules_fail_missing_or_unreadable_required_camera_modalities():
    result = evaluate_quality_report(
        {
            "modality_presence": {
                "videos:left": {"exists": False, "required": True},
                "timestamps:left": {"exists": True, "required": True},
            },
            "camera_readability": {
                "left": {"readable": False, "error": "Missing video"},
            },
            "timestamp_readability": {
                "left": {"readable": True},
            },
            "timestamp_alignment": {},
            "timestamp_duplicates": {},
            "imu": None,
            "video_infos": {},
            "fps": 30,
        }
    )

    assert "modality_presence:videos:left" in result["failed_checks"]
    assert "camera_readability:left" in result["failed_checks"]
    assert result["passed"] is False


def test_rules_fail_timestamp_alignment_from_configured_rate_threshold():
    result = evaluate_quality_report(
        {
            "source_session": "/data/session",
            "timestamp_alignment": {
                "left": {"over_tolerance_count": 0, "over_tolerance_rate": 0.0},
                "right": {"over_tolerance_count": 2, "over_tolerance_rate": 0.02},
            },
            "timestamp_duplicates": {},
            "imu": {"duplicate_count": 0},
            "video_infos": {},
            "fps": 30,
        },
        {
            "timestamp_alignment": {
                "enabled": True,
                "level": "fail",
                "max_over_tolerance_rate": 0.001,
            }
        },
    )

    assert result["passed"] is False
    assert result["failed_checks"] == ["timestamp_alignment:right"]
    assert result["checks"] == [
        {
            "name": "timestamp_alignment",
            "level": "fail",
            "passed": False,
            "stream": "right",
            "value": 0.02,
            "threshold": 0.001,
            "count": 2,
            "message": "right timestamp over-tolerance rate is 0.020000",
        }
    ]


def test_rules_warn_timestamp_duplicates_by_rate_threshold():
    result = evaluate_quality_report(
        {
            "source_session": "/data/session",
            "timestamp_alignment": {"left": {"over_tolerance_count": 0}},
            "timestamp_duplicates": {
                "left": {"duplicate_count": 2, "duplicate_rate": 0.02}
            },
            "imu": {"duplicate_count": 0, "duplicate_rate": 0.0},
            "video_infos": {},
            "fps": 30,
        },
        {
            "timestamp_duplicates": {
                "enabled": True,
                "level": "warning",
                "max_duplicate_rate": 0.001,
            }
        },
    )

    assert result["passed"] is True
    assert result["warnings"] == ["timestamp_duplicates:left"]
    assert result["checks"][0]["level"] == "warning"
    assert result["checks"][0]["value"] == 0.02


def test_rules_fail_imu_missing_duplicates_and_signal_anomalies():
    missing_result = evaluate_quality_report(
        {
            "timestamp_alignment": {},
            "timestamp_duplicates": {},
            "imu": None,
            "video_infos": {},
            "fps": 30,
        }
    )

    assert "imu_missing" in missing_result["failed_checks"]
    assert missing_result["passed"] is False

    anomalous_result = evaluate_quality_report(
        {
            "timestamp_alignment": {},
            "timestamp_duplicates": {},
            "imu": {
                "duplicate_rate": 0.02,
                "signal_quality": {
                    "all_zero_columns": ["gyro_x"],
                    "frozen_columns": ["accel_y"],
                    "jump_columns": ["accel_z"],
                },
            },
            "video_infos": {},
            "fps": 30,
        }
    )

    assert anomalous_result["passed"] is False
    assert anomalous_result["failed_checks"] == [
        "imu_duplicates",
        "imu_all_zero",
        "imu_frozen",
        "imu_jump",
    ]


def test_rules_fail_frame_count_consistency_over_ten_frames():
    result = evaluate_quality_report(
        {
            "timestamp_alignment": {},
            "timestamp_duplicates": {},
            "frame_count_consistency": {
                "max_video_frame_count_delta": 11,
                "max_timestamp_frame_count_delta": 0,
                "max_video_timestamp_count_delta": 1,
            },
            "imu": {"duplicate_rate": 0.0, "signal_quality": {}},
            "video_infos": {},
            "fps": 30,
        }
    )

    assert result["passed"] is False
    assert result["failed_checks"] == ["frame_count_consistency"]
    assert result["checks"][0]["value"] == 11
    assert result["checks"][0]["threshold"] == 10


def test_rules_fail_imu_time_coverage_when_gap_exceeds_ten_frames():
    result = evaluate_quality_report(
        {
            "timestamp_alignment": {},
            "timestamp_duplicates": {},
            "imu_time_coverage": {
                "computed": True,
                "max_missing_gap_frames": 12.5,
            },
            "imu": {"duplicate_rate": 0.0, "signal_quality": {}},
            "video_infos": {},
            "fps": 30,
        }
    )

    assert result["passed"] is False
    assert result["failed_checks"] == ["imu_time_coverage"]
    assert result["checks"][0]["value"] == 12.5
    assert result["checks"][0]["threshold"] == 10


def test_rules_can_disable_a_check():
    result = evaluate_quality_report(
        {
            "timestamp_alignment": {"right": {"over_tolerance_count": 5}},
            "timestamp_duplicates": {},
            "imu": {"duplicate_count": 0},
            "video_infos": {},
            "fps": 30,
        },
        {"timestamp_alignment": {"enabled": False}},
    )

    assert result["passed"] is True
    assert result["checks"] == []
