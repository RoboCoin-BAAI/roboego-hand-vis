"""Versioned checkpoints and conservative identities for stage-level recovery."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


SCHEMA_VERSION = 1


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.state-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked_path(root, relative):
    """Reject traversal and symlinks before reading or deleting owned artifacts."""
    root, relative = Path(root), Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Unsafe work path: {relative}')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Refusing symlinked work path: {current}')
    return current


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.absolute()), 'target': str(path.resolve()),
            'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def tree_identity(root):
    root = Path(root)
    if not root.exists():
        return None
    files = {}

    def visit(directory, ancestors):
        target = directory.resolve()
        if target in ancestors:
            raise ValueError(f'Directory symlink cycle: {directory}')
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                visit(path, ancestors | {target})
            elif path.is_file():
                files[str(path.relative_to(root))] = file_identity(path)

    visit(root, set())
    return files


def source_identity(root, *, scope=None, exclude=()):
    root = Path(root)
    result = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        capture_output=True, check=False)
    if result.returncode:
        paths = [path.relative_to(root) for path in root.rglob('*') if path.is_file()]
        commit = None
    else:
        paths = [Path(os.fsdecode(path)) for path in result.stdout.split(b'\0') if path]
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    files = {}
    for relative in sorted(set(paths)):
        name = relative.as_posix()
        if name in exclude or (scope and not name.startswith(scope)):
            continue
        if relative.suffix not in {'.py', '.sh', '.yaml', '.yml', '.toml', '.json', '.cfg', '.ini'}:
            continue
        path = root / relative
        files[name] = digest_file(path) if path.is_file() else 'missing'
    return {'commit': commit, 'files': files}


def fingerprint(plan):
    env = plan['environment']
    # Large inputs use filesystem identity, not a full multi-gigabyte content scan.
    identity = {
        'plan': plan,
        'video': file_identity(plan['video']),
        'checkpoint': file_identity(plan['checkpoint']),
        'depth': tree_identity(plan['depth']),
        'model_assets': {str(path): tree_identity(path) for path in (
            Path(env['EGOHANDKIT_ROOT']) / '_DATA',
            Path(env['MINT_ROOT']) / 'assets/mano',
            Path(env['MINT_ROOT']) / 'third_party/HaWoR/_DATA',
        )},
        'interpreters': [file_identity(plan['stages'][0]['command'][0]),
                         file_identity(env['EGOHANDKIT_PYTHON'])],
        'sources': {
            'project': source_identity(Path(__file__).resolve().parents[2],
                                       scope='roboego_hand_vis/hand/'),
            # The wrapper can create this exact copy; preflight checks equality.
            'mint': source_identity(env['MINT_ROOT'], exclude=('scripts/export_egohandkit_cache.py',)),
            'egohandkit': source_identity(env['EGOHANDKIT_ROOT']),
        },
        'environment': {key: os.environ.get(key) for key in
                        ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'PYOPENGL_PLATFORM')},
    }
    encoded = json.dumps(identity, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_state(path):
    state = json.loads(Path(path).read_text())
    if state.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported hand checkpoint version; use a new output directory')
    return state
