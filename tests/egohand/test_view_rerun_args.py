import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.view_ropedia_hamer_depth_alignment_rerun import parse_args


def test_parse_new_method_only_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["view", "--new_method_only"])

    args = parse_args()

    assert args.new_method_only is True
