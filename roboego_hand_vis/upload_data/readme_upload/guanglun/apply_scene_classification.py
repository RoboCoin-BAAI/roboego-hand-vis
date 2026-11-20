#!/usr/bin/env python3
"""Apply the frozen scene manifest, changing only the two existing JSON values."""
import argparse
import copy
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path


def object_value_spans(text, start=0):
    """Locate direct object values without matching similarly named nested keys."""
    decoder = json.JSONDecoder()
    index = start
    while text[index].isspace():
        index += 1
    if text[index] != '{':
        raise ValueError('Expected JSON object')
    index += 1
    spans = {}
    while True:
        while text[index].isspace():
            index += 1
        if text[index] == '}':
            return spans
        key, index = decoder.raw_decode(text, index)
        if not isinstance(key, str) or key in spans:
            raise ValueError('Invalid or duplicate JSON key')
        while text[index].isspace():
            index += 1
        if text[index] != ':':
            raise ValueError('Expected colon')
        index += 1
        while text[index].isspace():
            index += 1
        value_start = index
        _, index = decoder.raw_decode(text, index)
        spans[key] = (value_start, index)
        while text[index].isspace():
            index += 1
        if text[index] == ',':
            index += 1
        elif text[index] != '}':
            raise ValueError('Expected comma or closing brace')


def patch(raw, item):
    text = raw.decode('utf-8-sig')
    before = json.loads(text)
    scene = before.get('scene_classification')
    if not isinstance(scene, dict) or not {'category', 'area'} <= scene.keys():
        raise ValueError('Cannot add fields: existing scene_classification.category/area are required')
    if before.get('task_description') != item['task_description'] or scene.get('original_description') != item['original_description']:
        raise ValueError('Descriptions changed since classification; refusing stale decision')
    root_spans = object_value_spans(text)
    scene_spans = object_value_spans(text, root_spans['scene_classification'][0])
    expected = copy.deepcopy(before)
    expected['scene_classification']['category'] = item['category']
    expected['scene_classification']['area'] = item['area']
    for start, end, replacement in sorted([
        (*scene_spans[key], json.dumps(item[key], ensure_ascii=False))
        for key in ('category', 'area')], reverse=True):
        text = text[:start] + replacement + text[end:]
    if json.loads(text) != expected:
        raise ValueError('Unexpected JSON change outside category/area')
    return (b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b'') + text.encode('utf-8')


def atomic_write(path, raw, mode):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    root = args.root.resolve()
    report_dir = args.report_dir.resolve()
    if report_dir == root or root in report_dir.parents:
        parser.error('Report/backup directory must be outside the dataset')
    prepared, errors, seen = [], [], set()
    for item in manifest:
        relative = item['path']
        if not re.fullmatch(r'batch_\d{7}/[0-9a-f]{64}/annotation/episode_000000_structured\.json', relative):
            raise ValueError('Invalid manifest path: ' + relative)
        if not 1 <= int(relative.split('/')[0][6:]) <= 39 or relative in seen:
            raise ValueError('Out-of-range or duplicate path: ' + relative)
        seen.add(relative)
        path = root / relative
        try:
            if path.is_symlink() or root not in path.resolve().parents:
                raise ValueError('Unsafe annotation path')
            raw = path.read_bytes()
            revised = patch(raw, item)
            prepared.append((item, path, raw, revised, stat.S_IMODE(path.stat().st_mode)))
        except Exception as exc:
            errors.append(dict(path=relative, error=str(exc)))
    summary = dict(mode='apply' if args.apply else 'dry_run', total=len(manifest),
                   checked=len(prepared), changed=sum(raw != new for _, _, raw, new, _ in prepared),
                   unchanged=sum(raw == new for _, _, raw, new, _ in prepared), errors=errors)
    print(json.dumps(summary), flush=True)
    if not args.apply:
        return 2 if errors else 0
    if errors:
        # A clean preflight is required: no partial migration for invalid input.
        return 2
    report_dir.mkdir(parents=True, exist_ok=True)
    applied = []
    for index, (item, path, raw, revised, mode) in enumerate(prepared, 1):
        backup = report_dir / 'original_annotations' / item['path']
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise FileExistsError('Backup already exists; use a new report directory: ' + str(backup))
        with backup.open('xb') as stream:
            stream.write(raw)
        if path.read_bytes() != raw:
            raise RuntimeError('Annotation changed concurrently: ' + str(path))
        if raw != revised:
            atomic_write(path, revised, mode)
        actual = path.read_bytes()
        if actual != revised:
            raise RuntimeError('Write-back verification failed: ' + str(path))
        applied.append(dict(path=item['path'], changed=raw != revised,
                            before_sha256=hashlib.sha256(raw).hexdigest(),
                            after_sha256=hashlib.sha256(actual).hexdigest()))
        if index % 250 == 0:
            print(json.dumps(dict(verified=index, total=len(prepared))), flush=True)
    summary.update(verified=len(applied), only_category_area_changed=True,
                   report_dir=str(report_dir))
    (report_dir / 'apply_report.json').write_text(
        json.dumps(dict(summary=summary, files=applied), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
