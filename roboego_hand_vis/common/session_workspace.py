"""Use local session directories and copy stage outputs back, without editing files."""

import os
import shutil
import tempfile
from pathlib import Path

import yaml

from .session_paths import validate_relative_path

DEFAULT_OUTPUTS = Path(__file__).parents[1] / 'database/config/stage_outputs.yaml'


def _inside(root, path):
    path = Path(path)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Path outside root: {path}')
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f'Symlink is not allowed: {parent}')
    return path


class SessionWorkspace:
    def __init__(self, shared_root, local_root, stage, *, copy_missing=False,
                 outputs_config=DEFAULT_OUTPUTS):
        self.shared_root = Path(shared_root).expanduser().resolve()
        self.local_root = Path(local_root).expanduser().resolve()
        if not self.shared_root.is_dir():
            raise ValueError('Shared root must be an existing directory')
        if self.local_root.is_relative_to(self.shared_root) or self.shared_root.is_relative_to(self.local_root):
            raise ValueError('共享目录和本地目录不能相同或嵌套')
        self.stage, self.copy_missing = stage, copy_missing
        self.rules = yaml.safe_load(Path(outputs_config).read_text())['stages'][stage]
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.active_session = None

    def prepare(self, relative):
        relative = validate_relative_path(relative)
        source = _inside(self.shared_root, self.shared_root / relative)
        target = _inside(self.local_root, self.local_root / relative)
        if not target.exists() and self.copy_missing:
            if not source.is_dir():
                raise ValueError(f'Shared session does not exist: {relative}')
            # Copy to a temporary directory so an interrupted copy cannot look complete.
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix='.session-copy-', dir=target.parent))
            try:
                for path in source.rglob('*'):
                    _inside(source, path)
                shutil.copytree(source, temporary, dirs_exist_ok=True)
                os.rename(temporary, target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        if not target.is_dir():
            raise ValueError(f'Local session is missing: {relative}；请先运行基础质检复制数据，或手动准备本地目录')
        self.active_session, self.shared_session = target, source
        return target

    def publish(self, status):
        if self.active_session is None:
            raise RuntimeError('No active session')
        session, shared = self.active_session, self.shared_session
        files = set()
        # paths documents stage outputs; only required selects what to copy back.
        for pattern in self.rules['required']:
            validate_relative_path(pattern)
            matched = set()
            for path in session.glob(pattern):
                _inside(session, path)
                for item in path.rglob('*') if path.is_dir() else [path]:
                    _inside(session, item)
                    if item.is_file():
                        matched.add(item.relative_to(session))
            if status in ('completed', 'block') and not matched:
                raise ValueError(f'Missing stage output: {pattern}')
            files.update(matched)
        for relative in sorted(files):
            destination = _inside(self.shared_root, shared / relative)
            self._copy_file(session / relative, destination)
        return [str(relative) for relative in sorted(files)]

    def _copy_file(self, source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.stage-copy-', dir=destination.parent)
        os.close(fd)
        try:
            shutil.copy2(source, name)
            os.replace(name, destination)
        finally:
            Path(name).unlink(missing_ok=True)

    def release(self):
        self.active_session = None
