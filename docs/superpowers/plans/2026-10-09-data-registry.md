# PostgreSQL data service implementation plan

**Goal:** Provide three PostgreSQL tables for shared-directory downloads, each
with exactly `id`, `ks3_path`, `download_status`, and `local_path`.

**Architecture:** Explicit SQLAlchemy ORM models in `model.py` define `l1data`,
`l2data`, and `l3data`. `database.py` owns connection pools, sessions, transaction
boundaries, and initialization. `data_service.py` owns database operations;
`validators.py` owns input validation. Neither model nor utilities depend on the service.
Each table has an independent identity primary key and unique KS3 path. The only
statuses are `pending`, `processing`, `completed`, and `failed`. Short transactions
claim pending rows using `FOR UPDATE SKIP LOCKED`; network downloads run outside
transactions. No automatic expiry or reassignment of active downloads is offered
without ownership fields. Operators must stop the previous worker before reset.

**Tech stack:** PostgreSQL, SQLAlchemy 2, psycopg 3, pytest; system-installed PostgreSQL.

## Execution

- [x] Add `tests/database/test_data_service.py` using a temporary PostgreSQL schema
  selected through `TEST_DATABASE_URL`. Verify exactly four columns, independent
  IDs, unique registration under contention, one winner for a specific record,
  disjoint queue claims, skipped locked rows, completion, failure/retry, and
  invalid paths/statuses. Run before implementation and confirm failure.
- [x] Add `roboego_hand_vis/database/{__init__,model,database,data_service,validators,__main__}.py` with
  schema initialization and `register`, `get`, `claim`, `complete`, `fail`, and
  `retry` operations. All calls use independent short transactions; clients
  create their own connection pool per process. Reject unknown table names.
- [x] Add an optional `database` dependency group in `pyproject.toml` and update
  `uv.lock`. Use the existing system PostgreSQL service. Document one-time
  role/database creation and per-terminal connection configuration. The user
  removed the Docker configuration; it is no longer part of deployment.
- [x] Add Chinese setup and API documentation in
  `roboego_hand_vis/database/README.md`, including shared-path semantics,
  temporary-file publishing, manual recovery, and future column migrations.
- [x] Run the PostgreSQL integration suite, validate documented shell commands,
  run CLI initialization twice, and review changes. Stop the temporary test
  PostgreSQL instance after verification. Leave changes reviewable in the
  working tree without creating a commit.

## Verification commands

```bash
TEST_DATABASE_URL='postgresql+psycopg://postgres@/postgres?host=/path/to/test/socket&port=55439' python -m pytest tests/database -q
python -m roboego_hand_vis.database --help
git diff --check
```

Use only a disposable PostgreSQL instance for integration tests. Each test creates
and drops its own randomly named schema, without touching existing tables.

## Verification results

- PostgreSQL 14 integration tests: 18 passed, using isolated test schemas,
  including ORM reads/writes, Session commit, and rollback after exceptions.
- CLI initialization executed twice successfully against the disposable instance.
- `uv lock --check` and `git diff --check` passed.
- 2026-10-10: deployment instructions standardized on system PostgreSQL;
  removed obsolete Docker commands after the user deleted the Compose file.
- Independent code review found no important actionable issues.
- No production database service was started; deployment steps are documented.
