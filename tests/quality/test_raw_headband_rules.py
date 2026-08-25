from roboego_hand_vis.quality.raw_headband_rules import evaluate_quality_report


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
