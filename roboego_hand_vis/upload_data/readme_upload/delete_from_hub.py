#!/usr/bin/env python3
"""Delete explicitly named remote dataset repositories; never deletes local files."""
import argparse
import getpass
import re
import sys
import time

if __package__:
    from ._common import describe_error, get_token
else:
    from _common import describe_error, get_token


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub', choices=['hf', 'huggingface', 'ms', 'modelscope', 'both'], required=True)
    parser.add_argument('--repo-id', nargs='+', required=True, help='Full dataset IDs: owner/name')
    parser.add_argument('--dry-run', action='store_true', help='Show targets without credentials or network')
    args = parser.parse_args(argv)
    args.repo_id = list(dict.fromkeys(args.repo_id))
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo)
           or any(part in ('.', '..') for part in repo.split('/')) for repo in args.repo_id):
        parser.error('Use full dataset IDs (owner/name), not URLs or local paths.')
    args.hubs = ['huggingface', 'modelscope'] if args.hub == 'both' else [
        {'hf':'huggingface', 'ms':'modelscope'}.get(args.hub, args.hub)]
    return args


def make_delete_client(hub, token):
    if hub == 'huggingface':
        from huggingface_hub import HfApi
        return HfApi(token=token)
    from modelscope_hub import HubApi
    return HubApi(token=token)


def delete_and_verify(client, repo):
    # Do not retry a destructive call after an ambiguous response.
    client.delete_repo(repo_id=repo, repo_type='dataset')
    for attempt in range(3):
        if not client.repo_exists(repo_id=repo, repo_type='dataset'):
            return
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError('Deletion returned but repository still exists')


def main(argv=None):
    args = parse_args(argv)
    for hub in args.hubs:
        for repo in args.repo_id:
            print(f'Delete target: {hub} {repo}')
    if args.dry_run:
        print('Preview only. No requests sent.')
        return 0
    tokens = {}
    clients = {}
    # Obtain all credentials and construct clients before any deletion starts.
    try:
        for hub in args.hubs:
            token = get_token(hub) or getpass.getpass(f'{hub} token (hidden): ').strip()
            if not token:
                raise ValueError('A token is required for each selected platform')
            tokens[hub] = token
            clients[hub] = make_delete_client(hub, token)
    except (Exception, KeyboardInterrupt, EOFError) as error:
        print(f'Credential setup failed ({type(error).__name__}); no deletions attempted.', file=sys.stderr)
        return 1
    failed = 0
    for hub in args.hubs:
        for repo in args.repo_id:
            try:
                delete_and_verify(clients[hub], repo)
                print(f'Deleted and verified: {hub} {repo}')
            except Exception as error:
                # Both platform credentials must be redacted from diagnostics.
                message = describe_error(error, tokens[hub])
                for secret in tokens.values():
                    message = message.replace(secret, '[REDACTED]')
                print(f'Delete failed or unverified: {hub} {repo}: {message}', file=sys.stderr)
                failed += 1
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
