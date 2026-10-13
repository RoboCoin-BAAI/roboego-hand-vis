"""Safe shared-root-relative session paths, independent of database dependencies."""

from pathlib import Path, PurePosixPath, PureWindowsPath


def validate_relative_path(path: str) -> str:
    if (not path.strip() or "\x00" in path or "\\" in path
            or PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive
            or ".." in PurePosixPath(path).parts or str(PurePosixPath(path)) == "."):
        raise ValueError("local_path must be a relative path under the shared root")
    return str(PurePosixPath(path))


def resolve_session(root: Path, local_path: str) -> Path:
    session = (root / validate_relative_path(local_path)).resolve()
    if not session.is_relative_to(root.resolve()) or not session.is_dir():
        raise ValueError("Session is missing or outside the shared root")
    return session
