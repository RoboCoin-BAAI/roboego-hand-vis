"""Fill missing calibration from the same L3 device and headband model."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def device(session):
    match = re.fullmatch(r'(L3-\d{3})-\d{4}(?:-\d{2}){5}', Path(session).name)
    if not match:
        raise ValueError(f'Cannot identify L3 device: {session}')
    return match[1]


def model(session):
    rows = read_rows(Path(session) / 'meta/collect_info.jsonl')
    values = {r.get('hardware', {}).get('ego_headband') for r in rows}
    if len(values) != 1 or not all(isinstance(v, str) and v.strip() for v in values):
        raise ValueError('Missing or inconsistent ego_headband model')
    identifier = device(session)
    if any(r.get('collector_id') != identifier for r in rows):
        raise ValueError('collector_id does not match session device')
    return values.pop()


def repair_missing(ctx, donors):
    from .compute import check_calibration
    from ..common import Context

    destination = ctx.root / 'meta/calibrations.jsonl'
    if destination.exists():
        return dict(status='existing')
    identifier, headband = device(ctx.root), model(ctx.root)
    candidates = []
    for donor in donors:
        donor = Path(donor).resolve()
        if donor == ctx.root:
            continue
        try:
            if device(donor) == identifier and model(donor) == headband:
                candidates.append(donor)
        except (OSError, ValueError, TypeError):
            continue
    # Prefer the same batch, then the nearest session time (fixed-width timestamps).
    candidates.sort(key=lambda p: (p.parent != ctx.root.parent,
                                  abs(int(re.sub(r'\D', '', p.name[7:])) -
                                      int(re.sub(r'\D', '', ctx.root.name[7:]))), str(p)))
    for donor in candidates:
        try:
            donor_ctx = Context(donor)
            if not all(c['passed'] for c in check_calibration(donor_ctx)):
                continue
            rows = read_rows(donor / 'meta/calibrations.jsonl')
            payloads = [r['calibrations'] for r in rows]
            if not payloads or any(p != payloads[0] for p in payloads):
                continue  # An episode-dependent donor is ambiguous.
            repaired = []
            for episode in ctx.episodes:
                repaired.append(dict(episode_index=int(episode.name.removeprefix('episode_')),
                                     session=ctx.root.name[7:], calibrations=copy.deepcopy(payloads[0])))
            receipt = dict(status='copied', donor=str(donor), device=identifier, headband_model=headband,
                           created_at=datetime.now(timezone.utc).isoformat())
            # Return repair details to the final QC report; no separate receipt file.
            tmp = destination.with_name(f'.{destination.name}.{uuid4().hex}.tmp')
            try:
                tmp.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                       for row in repaired), encoding='utf-8')
                os.link(tmp, destination)  # Atomic publication; fails if destination already exists.
            finally:
                tmp.unlink(missing_ok=True)
            return receipt
        except FileExistsError:
            return dict(status='existing')
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return dict(status='no_matching_donor', device=identifier, headband_model=headband)
