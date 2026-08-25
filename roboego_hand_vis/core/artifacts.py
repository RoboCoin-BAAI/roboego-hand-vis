"""Small artifact descriptors used in stage reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Artifact:
    """Description of a file or directory produced by a pipeline stage."""

    name: str
    path: str | Path
    kind: str = "file"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "path": str(self.path),
            "kind": self.kind,
            "metadata": self.metadata,
        }
