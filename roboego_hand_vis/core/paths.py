"""Path helpers for project-relative pipeline code."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for this package layout."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate
