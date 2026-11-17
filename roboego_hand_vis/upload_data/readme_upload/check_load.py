#!/usr/bin/env python3
"""Independently smoke-test a local HumanCOIN dataset; never writes README or uploads."""
import argparse
import json
from pathlib import Path
import sys
if __package__:
    from ._common import repo_id_for
else:
    from _common import repo_id_for


def validate_dataset_load(folder, repo_id, info, loader=None):
    """Read first, middle and last samples using the HumanCOIN loader."""
    if loader is None:
        try:
            from humancoin import HumanCOINDataset
        except ImportError as error:
            raise RuntimeError('Install the loader first: pip install -U "humancoin>=0.1.1"') from error
        loader = HumanCOINDataset
    index = None
    try:
        dataset = loader(repo_id=repo_id, root=folder, video_backend='pyav')
        length = len(dataset)
        if length <= 0:
            raise ValueError('Dataset contains no readable master frames')
        if info.get('total_frames') is not None and length != info['total_frames']:
            raise ValueError('Loaded frame count differs from meta/info.json total_frames')
        indices = sorted({0, length // 2, length - 1})
        expected = {key for key in info['features'] if key.startswith('observation.')}
        for index in indices:
            sample = dataset[index]
            if not isinstance(sample, dict) or 'timestamp' not in sample:
                raise ValueError(f'Invalid loaded sample at frame {index}')
            missing = expected - sample.keys()
            if missing:
                raise ValueError(f'Missing features at frame {index}: {", ".join(sorted(missing))}')
        print(f'HumanCOIN load check passed: {length} frames; read samples {indices}')
    except Exception as error:
        raise RuntimeError(f'HumanCOIN load check failed at frame {index} ({type(error).__name__}): {error}') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--namespace', default='HumanCOIN')
    args = parser.parse_args(argv)
    try:
        folder = args.dataset.expanduser().resolve()
        if not folder.is_dir():
            raise ValueError('Dataset directory does not exist')
        info = json.loads((folder/'meta/info.json').read_text(encoding='utf-8'))
        if not isinstance(info, dict) or not isinstance(info.get('features'), dict) or not info['features']:
            raise ValueError('meta/info.json must contain feature definitions')
        validate_dataset_load(folder, repo_id_for(folder, args.namespace), info)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f'Load check failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
