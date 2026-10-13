"""Path validation independent of database models and connections."""

from roboego_hand_vis.common.session_paths import validate_relative_path, resolve_session


def validate_ks3_path(path: str) -> None:
    """Reject empty object paths and NUL characters before executing SQL."""
    if not path.strip() or "\x00" in path:
        raise ValueError("ks3_path must not be empty or contain NUL")
