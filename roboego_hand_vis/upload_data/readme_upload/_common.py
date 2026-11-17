"""Shared local-file selection, repository naming and credential diagnostics."""
import fnmatch
import json
import os
from pathlib import Path
import re

DEFAULT_IGNORE = [
    '.git/*', '*/.git/*', '.cache/*', '*/.cache/*',
    '__pycache__/*', '*/__pycache__/*', '*.pyc',
    'backup/*', '*/backup/*', '.env', '.env.*', '*/.env', '*/.env.*',
]


def repo_id_for(folder: Path, namespace: str, repo_id: str | None = None) -> str:
    name = folder.name
    metadata = folder / 'meta' / 'readme.json'
    if repo_id is None and metadata.is_file():
        name = json.loads(metadata.read_text(encoding='utf-8')).get('dataset_name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('meta/readme.json must contain a nonempty dataset_name')
    value = repo_id or f'{namespace}/{name}'
    parts = value.split('/')
    if len(parts) != 2 or any(
        not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', p) or '..' in p
        for p in parts
    ):
        raise ValueError('Repository must be owner/name with valid name components')
    return value


def plan_files(folder: Path, readme_only=False, allow=None, ignore=None) -> list[str]:
    """Return relative POSIX paths shared by preview and upload selection."""
    if not folder.is_dir():
        raise ValueError(f'Dataset directory does not exist: {folder}')
    excludes = DEFAULT_IGNORE + list(ignore or [])
    selected = []
    for path in sorted(folder.rglob('*')):
        rel = path.relative_to(folder).as_posix()
        if any(fnmatch.fnmatchcase(rel, p) for p in excludes):
            continue
        if readme_only and rel != 'README.md':
            continue
        if allow and not any(fnmatch.fnmatchcase(rel, p) for p in allow):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlinks are not supported; use regular files or hardlinks: {rel}')
        if path.is_file():
            selected.append(rel)
    if not selected:
        raise ValueError(f'No files selected in {folder}')
    return selected


def get_token(hub: str, explicit: str | None = None) -> str | None:
    names = ('HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGING_FACE_HUB_TOKEN') if hub == 'huggingface' else ('MODELSCOPE_TOKEN', 'MS_TOKEN', 'MODELSCOPE_API_TOKEN')
    return explicit or next((os.environ[n] for n in names if os.environ.get(n)), None)



def describe_error(exc, token=None):
    """Expose HTTP diagnostics without printing requests, headers or tokens."""
    parts = [type(exc).__name__]
    response = getattr(exc, 'response', None)
    if response is not None:
        status = getattr(response, 'status_code', None)
        if status is not None:
            parts.append(f'HTTP {status}')
        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = payload.get('error') or payload.get('message') or payload.get('Message')
                if isinstance(message, str):
                    parts.append(message)
        except (ValueError, TypeError):
            pass
        request_id = response.headers.get('x-request-id')
        if request_id:
            parts.append(f'Request ID: {request_id}')
    # Include neither arbitrary exception strings nor full HTTP response bodies.
    message = ' | '.join(parts)
    secrets = [token] + [value for key, value in os.environ.items()
                         if ('TOKEN' in key.upper() or 'SECRET' in key.upper()) and value]
    for secret in sorted((v for v in secrets if v), key=len, reverse=True):
        message = message.replace(secret, '[REDACTED]')
    message = re.sub(r'(?i)Bearer\s+[^\s,;]+', 'Bearer [REDACTED]', message)
    message = re.sub(r'\bhf_[A-Za-z0-9]+', '[REDACTED]', message)
    return ' '.join(message.split())[:1500]
