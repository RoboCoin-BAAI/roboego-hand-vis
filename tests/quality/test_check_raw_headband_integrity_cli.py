from scripts.quality.check_raw_headband_integrity import build_arg_parser


def test_check_raw_headband_integrity_cli_exposes_session_argument(tmp_path):
    session = tmp_path / "session_20260804_224442"

    args = build_arg_parser().parse_args(["--session", str(session)])

    assert args.session == str(session)


def test_check_raw_headband_integrity_cli_accepts_video_extension(tmp_path):
    session = tmp_path / "session_20260804_224442"

    args = build_arg_parser().parse_args(["--session", str(session), "--video_ext", ".mkv"])

    assert args.video_ext == ".mkv"
