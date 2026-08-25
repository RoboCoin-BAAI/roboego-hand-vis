"""Runtime wrappers for stages that still call existing scripts."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PythonRuntime:
    """A named Python interpreter used by a pipeline stage."""

    name: str
    python: str | Path
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | Path | None = None

    def command(self, args: Sequence[str | Path]) -> list[str]:
        """Build a command beginning with the configured interpreter."""
        return [str(self.python), *[str(arg) for arg in args]]

    def run(
        self,
        args: Sequence[str | Path],
        *,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run a Python command in this runtime."""
        return subprocess.run(
            self.command(args),
            check=check,
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=dict(self.env) if self.env else None,
            text=True,
            capture_output=capture_output,
        )
