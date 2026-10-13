import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest
pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a disposable PostgreSQL database")
    from roboego_hand_vis.database import DataService
    admin = create_engine(url)
    schema = "test_service_" + uuid4().hex
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    service = DataService(engine)
    service.initialize()
    try:
        yield service
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def test_schema_has_expected_columns_and_independent_ids(db):
    db.initialize()
    inspector = inspect(db.engine)
    assert set(inspector.get_table_names()) == {"l1data", "l2data", "l3data"}
    for table in ("L1data", "L2data", "L3data"):
        expected_columns = ["id", "ks3_path", "download_status", "local_path"]
        if table.lower() == "l1data":
            import yaml
            from pathlib import Path
            config = yaml.safe_load(Path("roboego_hand_vis/database/config/l1_fields.yaml").read_text())
            expected_columns += [stage["status_field"] for stage in config["stages"]]
        assert [c["name"] for c in inspector.get_columns(table.lower())] == expected_columns
        row = db.register(table, "ks3://bucket/sample")
        assert {key: row[key] for key in expected_columns[:4]} == dict(
            id=1, ks3_path="ks3://bucket/sample", download_status="pending", local_path=None
        )
        for field in expected_columns[4:]:
            assert row[field] == "pending"


def test_integrity_claim_requires_download_and_preserves_other_fields(db):
    pending = db.register("l1data", "ks3://bucket/pending")
    ready = db.register("l1data", "ks3://bucket/ready")
    assert db.claim_raw_headband_integrity() is None
    db.claim("l1data", ready["id"])
    db.complete("l1data", ready["id"], "ready")
    row = db.claim_raw_headband_integrity()
    assert row["id"] == ready["id"]
    assert row["raw_headband_integrity_status"] == "processing"
    assert db.claim_raw_headband_integrity() is None
    assert db.finish_raw_headband_integrity(ready["id"], passed=False)
    assert not db.finish_raw_headband_integrity(ready["id"], passed=True)
    final = db.get("l1data", ready["id"])
    assert final["raw_headband_integrity_status"] == "block"
    assert final["download_status"] == "completed"
    assert final["local_path"] == "ready"
    assert final["video_simple_quality_status"] == "pending"
    assert db.get("l1data", pending["id"]) == pending


def test_integrity_execution_failure_is_distinct_from_block(db):
    ready = db.register("l1data", "ks3://bucket/error")
    db.claim("l1data", ready["id"])
    db.complete("l1data", ready["id"], "error")
    assert db.claim_raw_headband_integrity()
    assert db.finish_raw_headband_integrity(ready["id"], passed=None)
    assert db.get("l1data", ready["id"])["raw_headband_integrity_status"] == "failed"


def test_all_status_columns_allow_block_after_repeatable_migration(db):
    from pathlib import Path
    migration = Path("roboego_hand_vis/database/migrations/003_unify_status_values.sql").read_text()
    for table in ("l1data", "l2data", "l3data"):
        db.register(table, f"ks3://bucket/{table}")
    for _ in range(2):
        with db.engine.begin() as conn:
            conn.execute(text(migration))
            for table in ("l1data", "l2data", "l3data"):
                fields = [c["name"] for c in inspect(conn).get_columns(table) if c["name"].endswith("_status")]
                assignments = ", ".join(f"{field}='block'" for field in fields)
                conn.execute(text(f"UPDATE {table} SET {assignments}"))
        for table in ("l1data", "l2data", "l3data"):
            row = db.get(table, 1)
            assert all(value == "block" for field, value in row.items() if field.endswith("_status"))


def test_new_schema_all_status_columns_allow_block(db):
    for table in ("l1data", "l2data", "l3data"):
        row = db.register(table, f"ks3://bucket/{table}")
        fields = [field for field in row if field.endswith("_status")]
        assignments = ", ".join(f"{field}='block'" for field in fields)
        with db.engine.begin() as conn:
            conn.execute(text(f"UPDATE {table} SET {assignments}"))
        assert all(db.get(table, row["id"])[field] == "block" for field in fields)


def test_video_gate_requires_previous_gate_pass_and_claims_once(db):
    for stage_status in ("pending", "processing", "block", "failed", "completed"):
        row = db.register("l1data", f"ks3://bucket/{stage_status}")
        db.claim("l1data", row["id"])
        db.complete("l1data", row["id"], stage_status)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET raw_headband_integrity_status=:status WHERE id=:id"),
                         dict(status=stage_status, id=row["id"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: db.claim_video_simple_quality(), range(8)))
    winners = [row for row in claims if row is not None]
    assert len(winners) == 1
    winner = winners[0]
    assert winner["local_path"] == "completed"
    assert winner["video_simple_quality_status"] == "processing"
    assert db.finish_video_simple_quality(winner["id"], passed=False)
    assert not db.finish_video_simple_quality(winner["id"], passed=True)
    final = db.get("l1data", winner["id"])
    assert final["video_simple_quality_status"] == "block"
    assert final["raw_headband_integrity_status"] == "completed"
    assert final["download_status"] == "completed"
    assert final["local_path"] == "completed"
    assert final["stereo_rectification_status"] == "pending"


def test_hand_detection_requires_gates_and_concurrent_claim_is_unique(db):
    for stage_status in ("pending", "processing", "block", "failed", "completed"):
        row = db.register("l1data", f"ks3://bucket/{stage_status}")
        db.claim("l1data", row["id"])
        db.complete("l1data", row["id"], stage_status)
        db.claim_raw_headband_integrity()
        db.finish_raw_headband_integrity(row["id"], passed=True)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET video_simple_quality_status=:status WHERE id=:id"),
                         dict(status=stage_status, id=row["id"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: db.claim_detect_hand(), range(8)))
    winners = [row for row in claims if row is not None]
    assert len(winners) == 1
    winner = winners[0]
    assert winner["local_path"] == "completed"
    assert winner["detect_hand_status"] == "processing"
    assert db.finish_detect_hand(winner["id"], succeeded=False)
    assert not db.finish_detect_hand(winner["id"], succeeded=True)
    final = db.get("l1data", winner["id"])
    assert final["detect_hand_status"] == "failed"
    assert final["video_simple_quality_status"] == "completed"
    assert final["raw_headband_integrity_status"] == "completed"
    assert final["download_status"] == "completed"
    assert final["local_path"] == "completed"
    assert final["stereo_rectification_status"] == "pending"


def test_rectification_requires_gates_but_not_hand_and_claims_once(db):
    for status in ("pending", "processing", "block", "failed", "completed"):
        row = db.register("l1data", f"ks3://bucket/{status}")
        db.claim("l1data", row["id"])
        db.complete("l1data", row["id"], status)
        db.claim_raw_headband_integrity()
        db.finish_raw_headband_integrity(row["id"], passed=True)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET video_simple_quality_status=:status WHERE id=:id"),
                         dict(status=status, id=row["id"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: db.claim_stereo_rectification(), range(8)))
    winners = [row for row in claims if row is not None]
    assert len(winners) == 1
    winner = winners[0]
    assert winner["local_path"] == "completed"
    assert winner["detect_hand_status"] == "pending"
    assert winner["stereo_rectification_status"] == "processing"
    assert db.finish_stereo_rectification(winner["id"], succeeded=False)
    assert not db.finish_stereo_rectification(winner["id"], succeeded=True)
    final = db.get("l1data", winner["id"])
    assert final["stereo_rectification_status"] == "failed"
    assert final["download_status"] == "completed"
    assert final["local_path"] == "completed"
    assert final["stereo_depth_gate_status"] == "pending"


def test_rectification_runner_releases_connection(db, tmp_path):
    import yaml
    from roboego_hand_vis.stereo_rectification.database_run import run_pending
    row = db.register("l1data", "ks3://bucket/session")
    assert db.claim_stereo_rectification() is None
    db.claim("l1data", row["id"])
    db.complete("l1data", row["id"], "session")
    db.claim_raw_headband_integrity()
    db.finish_raw_headband_integrity(row["id"], passed=True)
    db.claim_video_simple_quality()
    db.finish_video_simple_quality(row["id"], passed=True)
    session = tmp_path / "session"
    calibration = session / "calibrations"
    calibration.mkdir(parents=True)
    cam = dict(camera_model="pinhole", distortion_model="equidistant",
               intrinsics=[550, 550, 960, 540], distortion_coeffs=[0.1, 0, 0, 0])
    (calibration / "camera-camchain.yaml").write_text(yaml.safe_dump(dict(cam0=cam, cam1=cam)))

    def check(path, options):
        assert db.engine.pool.checkedout() == 0
        assert db.get("l1data", row["id"])["stereo_rectification_status"] == "processing"
        for rel in ("videos/left_rectified.mp4", "videos/right_rectified.mp4", "dataset_manifest/dataset_manifest.json"):
            target = path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"output")
        return {"status": "ok"}

    opts = dict(dataset_adapter="raw", model="equidistant", balance=0, width=1600,
                height=900, overwrite=False, dry_run=False)
    assert run_pending(db, tmp_path, opts, runner=check)[0]["status"] == "completed"
    assert db.get("l1data", row["id"])["stereo_rectification_status"] == "completed"


def test_qwen_requires_gates_but_not_rectification_and_claims_once(db):
    for status in ("pending", "processing", "block", "failed", "completed"):
        row = db.register("l1data", f"ks3://bucket/{status}")
        db.claim("l1data", row["id"])
        db.complete("l1data", row["id"], status)
        db.claim_raw_headband_integrity()
        db.finish_raw_headband_integrity(row["id"], passed=True)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET video_simple_quality_status=:status WHERE id=:id"),
                         dict(status=status, id=row["id"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: db.claim_qwen_annotation(), range(8)))
    winners = [row for row in claims if row is not None]
    assert len(winners) == 1
    winner = winners[0]
    assert winner["local_path"] == "completed"
    assert winner["stereo_rectification_status"] == "pending"
    assert winner["qwen_annotation_status"] == "processing"
    assert db.finish_qwen_annotation(winner["id"], succeeded=False)
    assert not db.finish_qwen_annotation(winner["id"], succeeded=True)
    final = db.get("l1data", winner["id"])
    assert final["qwen_annotation_status"] == "failed"
    assert final["download_status"] == "completed"
    assert final["local_path"] == "completed"
    assert final["post_annotation_actions_status"] == "pending"


def test_post_actions_requires_qwen_and_claims_once(db):
    for status in ("pending", "processing", "block", "failed", "completed"):
        row = db.register("l1data", f"ks3://bucket/{status}")
        db.claim("l1data", row["id"])
        db.complete("l1data", row["id"], status)
        db.claim_raw_headband_integrity()
        db.finish_raw_headband_integrity(row["id"], passed=True)
        db.claim_video_simple_quality()
        db.finish_video_simple_quality(row["id"], passed=True)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET qwen_annotation_status=:status WHERE id=:id"),
                         dict(status=status, id=row["id"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: db.claim_post_annotation_actions(), range(8)))
    winners = [row for row in claims if row is not None]
    assert len(winners) == 1
    winner = winners[0]
    assert winner["local_path"] == "completed"
    assert winner["post_annotation_actions_status"] == "processing"
    assert db.finish_post_annotation_actions(winner["id"], succeeded=False)
    assert not db.finish_post_annotation_actions(winner["id"], succeeded=True)
    final = db.get("l1data", winner["id"])
    assert final["post_annotation_actions_status"] == "failed"
    assert final["qwen_annotation_status"] == "completed"
    assert final["download_status"] == "completed"
    assert final["local_path"] == "completed"
    assert final["stereo_rectification_status"] == "pending"


def test_post_actions_runner_releases_connection(db, tmp_path, monkeypatch):
    import json
    from roboego_hand_vis.post_annotation_actions.database_run import run_pending
    row = db.register("l1data", "ks3://bucket/session")
    assert db.claim_post_annotation_actions() is None
    db.claim("l1data", row["id"])
    db.complete("l1data", row["id"], "session")
    db.claim_raw_headband_integrity()
    db.finish_raw_headband_integrity(row["id"], passed=True)
    db.claim_video_simple_quality()
    db.finish_video_simple_quality(row["id"], passed=True)
    db.claim_qwen_annotation()
    db.finish_qwen_annotation(row["id"], succeeded=True)
    chunks = tmp_path / "session" / "annotation" / "qwen_frame_annotation_chunks"
    chunks.mkdir(parents=True)
    (chunks / "chunk_0000.json").write_text(json.dumps({"frames": [{"t": 0}]}))
    monkeypatch.setenv("TEST_POST_KEY", "test-only")

    def check(cfg):
        assert db.engine.pool.checkedout() == 0
        assert db.get("l1data", row["id"])["post_annotation_actions_status"] == "processing"
        cfg.resolved_chunk_dir.mkdir(parents=True)
        (cfg.resolved_chunk_dir / "chunk_0000.json").write_text(json.dumps({"frames": [{"t": 0}]}))
        cfg.merged_output_json.write_text(json.dumps({"frame_count": 1, "frames": [{"t": 0}]}))
        return {"status": "complete", "merged_frame_count": 1}

    data = {"parameters": {"api_key_env": "TEST_POST_KEY", "generate_subtasks": False}}
    assert run_pending(db, tmp_path, data, runner=check)[0]["status"] == "completed"
    assert db.get("l1data", row["id"])["post_annotation_actions_status"] == "completed"


def test_qwen_runner_releases_connection(db, tmp_path, monkeypatch):
    import json
    from roboego_hand_vis.qwen_annotation.database_run import run_pending
    row = db.register("l1data", "ks3://bucket/session")
    assert db.claim_qwen_annotation() is None
    db.claim("l1data", row["id"])
    db.complete("l1data", row["id"], "session")
    db.claim_raw_headband_integrity()
    db.finish_raw_headband_integrity(row["id"], passed=True)
    db.claim_video_simple_quality()
    db.finish_video_simple_quality(row["id"], passed=True)
    videos = tmp_path / "session" / "videos"
    videos.mkdir(parents=True)
    (videos / "left.mkv").touch()
    prompt = tmp_path / "prompt.md"
    prompt.write_text("test prompt")
    monkeypatch.setenv("TEST_QWEN_KEY", "test-only")

    def check(cfg):
        assert db.engine.pool.checkedout() == 0
        assert db.get("l1data", row["id"])["qwen_annotation_status"] == "processing"
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_json.write_text(json.dumps(dict(status="complete", source_video=str(cfg.video),
            failed_batch_count=0, failed_batches=[], frames=[{"t": 0}])))
        return dict(status="complete", failed_batch_count=0, frame_count=1)

    data = {"parameters": {"prompt": str(prompt), "api_key_env": "TEST_QWEN_KEY"}}
    assert run_pending(db, tmp_path, data, runner=check)[0]["status"] == "completed"
    assert db.get("l1data", row["id"])["qwen_annotation_status"] == "completed"


def test_hand_runner_releases_connection_and_zero_presence_completes(db, tmp_path):
    from argparse import Namespace
    from roboego_hand_vis.detect_hand.cli import config_from_yaml
    from roboego_hand_vis.detect_hand.database_run import run_pending
    row = db.register("l1data", "ks3://bucket/session")
    assert db.claim_detect_hand() is None
    db.claim("l1data", row["id"])
    db.complete("l1data", row["id"], "session")
    db.claim_raw_headband_integrity()
    db.finish_raw_headband_integrity(row["id"], passed=True)
    db.claim_video_simple_quality()
    db.finish_video_simple_quality(row["id"], passed=True)
    videos = tmp_path / "session" / "videos"
    videos.mkdir(parents=True)
    (videos / "left.mkv").touch()
    model = tmp_path / "model.pt"
    model.touch()
    cfg = config_from_yaml({}, Namespace(video=str(videos / "left.mkv"), model=str(model),
        output_json="", sample_every=None, conf=None, device="cpu", imgsz=None, batch_size=None))

    def check(config):
        assert db.engine.pool.checkedout() == 0
        assert db.get("l1data", row["id"])["detect_hand_status"] == "processing"
        return {"left_hand_presence_percent": 0.0, "right_hand_presence_percent": 0.0}

    assert run_pending(db, tmp_path, cfg, runner=check)[0]["status"] == "completed"
    assert db.get("l1data", row["id"])["detect_hand_status"] == "completed"


@pytest.mark.parametrize("passed,expected", [(True, "completed"), (None, "failed")])
def test_video_gate_runner_releases_db_connection(db, tmp_path, passed, expected):
    from argparse import Namespace
    from roboego_hand_vis.video_simple_quality.cli import config_from_yaml
    from roboego_hand_vis.video_simple_quality.database_run import run_pending
    row = db.register("l1data", "ks3://bucket/session")
    assert db.claim_video_simple_quality() is None
    db.claim("l1data", row["id"])
    db.complete("l1data", row["id"], "session")
    db.claim_raw_headband_integrity()
    db.finish_raw_headband_integrity(row["id"], passed=True)
    (tmp_path / "session").mkdir()
    config = config_from_yaml({}, Namespace(session=str(tmp_path), video="", output_json=""))

    def check(cfg):
        assert db.engine.pool.checkedout() == 0
        assert db.get("l1data", row["id"])["video_simple_quality_status"] == "processing"
        if passed is None:
            raise RuntimeError("decode error")
        return {"passed": passed}

    assert run_pending(db, config, runner=check)[0]["status"] == expected
    assert db.get("l1data", row["id"])["video_simple_quality_status"] == expected


def test_integrity_block_migration_is_repeatable_and_preserves_rows(db):
    from pathlib import Path
    ready = db.register("l1data", "ks3://bucket/migration")
    old_constraint = "l1data_raw_headband_integrity_status_valid"
    with db.engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE l1data DROP CONSTRAINT {old_constraint}"))
        conn.execute(text(f"ALTER TABLE l1data ADD CONSTRAINT {old_constraint} CHECK "
                          "(raw_headband_integrity_status IN ('pending','processing','completed','failed'))"))
    migration = Path("roboego_hand_vis/database/migrations/002_allow_integrity_block.sql").read_text()
    for _ in range(2):
        with db.engine.begin() as conn:
            conn.execute(text(migration))
    assert db.get("l1data", ready["id"]) == ready
    db.claim("l1data", ready["id"])
    db.complete("l1data", ready["id"], "migration")
    db.claim_raw_headband_integrity()
    assert db.finish_raw_headband_integrity(ready["id"], passed=False)


def test_integrity_concurrent_workers_claim_once(db):
    ready = db.register("l1data", "ks3://bucket/ready")
    db.claim("l1data", ready["id"])
    db.complete("l1data", ready["id"], "ready")
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: db.claim_raw_headband_integrity(), range(8)))
    assert sum(row is not None for row in rows) == 1
    assert db.finish_raw_headband_integrity(ready["id"], passed=True)
    assert db.get("l1data", ready["id"])["raw_headband_integrity_status"] == "completed"


def test_integrity_runner_releases_connection_during_quality_check(db, tmp_path):
    from argparse import Namespace
    from roboego_hand_vis.raw_headband_integrity.cli import config_from_yaml
    from roboego_hand_vis.raw_headband_integrity.database_run import run_pending

    ready = db.register("l1data", "ks3://bucket/session")
    db.claim("l1data", ready["id"])
    db.complete("l1data", ready["id"], "session")
    (tmp_path / "session").mkdir()
    config = config_from_yaml({}, Namespace(
        session=str(tmp_path), output_root="", camera_stream=None, overwrite=True))

    def check(cfg):
        assert db.engine.pool.checkedout() == 0
        assert db.get("l1data", ready["id"])["raw_headband_integrity_status"] == "processing"
        return {"gate": {"passed": True}}

    assert run_pending(db, config, runner=check)[0]["status"] == "completed"


def test_concurrent_registration_preserves_existing_record(db):
    barrier = Barrier(8)

    def register(_):
        barrier.wait()
        return db.register("l1data", "ks3://bucket/same")

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(register, range(8)))
    assert len({row["id"] for row in rows}) == 1
    row = rows[0]
    db.claim("l1data", row["id"])
    db.complete("l1data", row["id"], "L1/sample.bin")
    assert db.register("l1data", row["ks3_path"])["download_status"] == "completed"


def test_only_one_worker_claims_specific_record(db):
    row = db.register("l1data", "ks3://bucket/specific")
    barrier = Barrier(8)

    def claim(_):
        barrier.wait()
        return db.claim("l1data", row["id"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(8)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0]["download_status"] == "processing"


def test_queue_claims_are_disjoint(db):
    ids = {db.register("l2data", f"ks3://bucket/{i}")["id"] for i in range(24)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: db.claim("l2data"), range(32)))
    claimed = [row["id"] for row in rows if row is not None]
    assert len(claimed) == len(ids)
    assert set(claimed) == ids


def test_queue_skips_locked_row(db):
    first = db.register("l3data", "ks3://bucket/first")
    second = db.register("l3data", "ks3://bucket/second")
    with db.engine.begin() as lock:
        lock.execute(text("SELECT id FROM l3data WHERE id=:id FOR UPDATE"), first)
        with ThreadPoolExecutor(max_workers=1) as pool:
            row = pool.submit(db.claim, "l3data").result(timeout=5)
        assert row["id"] == second["id"]
    assert db.claim("l3data")["id"] == first["id"]


def test_status_lifecycle_and_failed_retry(db):
    row = db.register("l1data", "ks3://bucket/lifecycle")
    assert db.get("l1data", row["id"]) == row
    assert db.get("l1data", -1) is None
    assert not db.complete("l1data", row["id"], "L1/sample")
    assert not db.fail("l1data", row["id"])
    assert db.claim("l1data", row["id"])
    assert not db.retry("l1data", row["id"])
    assert db.fail("l1data", row["id"])
    assert db.claim("l1data", row["id"]) is None
    assert db.retry("l1data", row["id"])
    assert db.claim("l1data", row["id"])
    assert db.complete("l1data", row["id"], "L1/sample")
    assert db.get("l1data", row["id"])["local_path"] == "L1/sample"
    assert db.get("l1data", row["id"])["download_status"] == "completed"
    assert not db.fail("l1data", row["id"])
    assert not db.retry("l1data", row["id"])


@pytest.mark.parametrize("path", ["", " ", "/data/a", "../a", "a/../b", "C:/a", "a\\b", "."])
def test_invalid_relative_paths_rejected(db, path):
    row = db.register("l1data", "ks3://bucket/path")
    db.claim("l1data", row["id"])
    with pytest.raises(ValueError):
        db.complete("l1data", row["id"], path)
    assert db.get("l1data", row["id"])["download_status"] == "processing"


def test_database_rejects_invalid_status_and_completion_without_path(db):
    row = db.register("l1data", "ks3://bucket/constraint")
    for status in ("downloading", "进行中", "unknown", "completed"):
        with pytest.raises(IntegrityError), db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET download_status=:status WHERE id=:id"),
                         dict(status=status, id=row["id"]))


def test_invalid_input_rejected(db):
    with pytest.raises(ValueError):
        db.register("l1data; DROP TABLE l2data", "ks3://bucket/a")
    with pytest.raises(ValueError):
        db.register("l1data", " ")


def test_orm_models_and_session_commit(db):
    from roboego_hand_vis.database.model import L1Data, L2Data, L3Data, TaskStatus

    for model in (L1Data, L2Data, L3Data):
        with db.database.with_session() as session:
            row = model(ks3_path="ks3://bucket/orm")
            session.add(row)
            session.flush()
            data_id = row.id
            assert row.download_status == TaskStatus.PENDING.value
        # The context committed; service operations see ORM-created records.
        assert db.get(model.__tablename__, data_id)["ks3_path"] == row.ks3_path
        assert db.claim(model.__tablename__, data_id)["download_status"] == TaskStatus.PROCESSING.value


def test_session_rolls_back_and_remains_usable(db):
    from roboego_hand_vis.database.model import L1Data

    with pytest.raises(RuntimeError, match="abort"):
        with db.database.with_session() as session:
            row = L1Data(ks3_path="ks3://bucket/rollback")
            session.add(row)
            session.flush()
            rolled_back_id = row.id
            raise RuntimeError("abort")
    assert db.get("l1data", rolled_back_id) is None

    existing = db.register("l1data", "ks3://bucket/duplicate")
    with pytest.raises(IntegrityError):
        with db.database.with_session() as session:
            session.add(L1Data(ks3_path=existing["ks3_path"]))
    assert db.get("l1data", existing["id"]) == existing
    assert db.register("l1data", "ks3://bucket/after-rollback")


def test_l1_stage_status_constraints(db):
    import yaml
    from pathlib import Path
    config = yaml.safe_load(Path("roboego_hand_vis/database/config/l1_fields.yaml").read_text())
    row = db.register("l1data", "ks3://bucket/stages")
    for stage in config["stages"]:
        field = stage["status_field"]
        for invalid in ("unknown", None):
            with pytest.raises(IntegrityError), db.engine.begin() as conn:
                conn.execute(text(f'UPDATE l1data SET "{field}"=:value WHERE id=:id'),
                             dict(value=invalid, id=row["id"]))
        for status in ("pending", "processing", "completed", "failed"):
            with db.engine.begin() as conn:
                conn.execute(text(f'UPDATE l1data SET "{field}"=:value WHERE id=:id'),
                             dict(value=status, id=row["id"]))
            assert db.get("l1data", row["id"])[field] == status


def test_l1_migration_preserves_old_data_and_is_repeatable(db):
    from pathlib import Path
    migration = Path("roboego_hand_vis/database/migrations/001_add_l1_stage_status.sql").read_text()
    with db.engine.begin() as conn:
        conn.execute(text("DROP TABLE l1data"))
        conn.execute(text("""CREATE TABLE l1data (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            ks3_path TEXT NOT NULL UNIQUE,
            download_status TEXT NOT NULL DEFAULT 'pending',
            local_path TEXT
        )"""))
        conn.execute(text("""INSERT INTO l1data (ks3_path, download_status, local_path)
            VALUES ('ks3://bucket/existing', 'completed', 'L1/existing')"""))
        conn.execute(text(migration))
        conn.execute(text(migration))
    row = db.get("l1data", 1)
    assert row["download_status"] == "completed"
    assert row["local_path"] == "L1/existing"
    assert len(row) == 16
    assert all(value == "pending" for key, value in row.items()
               if key.endswith("_status") and key != "download_status")


def test_registration_reports_new_or_existing_under_contention(db):
    barrier = Barrier(8)

    def register(_):
        barrier.wait()
        return db.register_if_absent("l1data", "ks3://bucket/new-or-existing")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(register, range(8)))
    assert sum(created for _, created in results) == 1
    assert len({row["id"] for row, _ in results}) == 1


def test_ingest_repeat_preserves_download_and_stage_states(db):
    from roboego_hand_vis.database.ingest import ingest_paths

    path = "ks3://bucket/session001"
    first = ingest_paths(db, "l1data", [path])[0]
    assert first["result"] == "created"
    db.claim("l1data", first["id"])
    db.complete("l1data", first["id"], "L1/session001")
    with db.engine.begin() as conn:
        conn.execute(text("UPDATE l1data SET qwen_annotation_status='completed' WHERE id=:id"), first)
    again = ingest_paths(db, "l1data", [path])[0]
    assert again["result"] == "existing"
    assert again["id"] == first["id"]
    row = db.get("l1data", first["id"])
    assert row["download_status"] == "completed"
    assert row["local_path"] == "L1/session001"
    assert row["qwen_annotation_status"] == "completed"


def test_ingest_invalid_path_does_not_partially_insert(db):
    from roboego_hand_vis.database.ingest import ingest_paths

    with pytest.raises(ValueError):
        ingest_paths(db, "l1data", ["ks3://bucket/valid", "local/file"])
    with db.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM l1data")).scalar_one() == 0


def test_ingest_cli_paths_and_file(db, tmp_path):
    import json
    import subprocess
    import sys

    with db.engine.connect() as conn:
        schema = conn.execute(text("SELECT current_schema()")).scalar_one()
    url = db.engine.url.update_query_dict({"options": f"-csearch_path={schema}"})
    env = dict(os.environ, DATABASE_URL=url.render_as_string(hide_password=False))
    input_file = tmp_path / "paths.txt"
    input_file.write_text("# session list\nks3://bucket/session002\n\n")
    command = [sys.executable, "-m", "roboego_hand_vis.database.ingest", "--table", "l1data",
               "--file", str(input_file), "ks3://bucket/session001"]
    for expected in ("created", "existing"):
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        assert len(output["records"]) == 2
        assert {row["result"] for row in output["records"]} == {expected}
        assert all(row["download_status"] == "pending" for row in output["records"])


def test_database_download_adapter_writes_status_without_holding_connection(db, tmp_path):
    from roboego_hand_vis.download_data.database_adapter import download_pending_l1
    from roboego_hand_vis.download_data.downloader import DownloadError, DownloadResult

    good = db.register("l1data", "ks3://bucket/data/good")
    bad = db.register("l1data", "ks3://bucket/data/bad")

    class Downloader:
        def download(self, source):
            assert db.engine.pool.checkedout() == 0
            if source.endswith("bad"):
                assert db.get("l1data", bad["id"])["download_status"] == "processing"
                raise DownloadError("test transfer failure")
            assert db.get("l1data", good["id"])["download_status"] == "processing"
            return DownloadResult(tmp_path / "good", "good")

    records = download_pending_l1(db, Downloader())
    assert [r["status"] for r in records] == ["completed", "failed"]
    assert db.get("l1data", good["id"])["local_path"] == "good"
    assert db.get("l1data", bad["id"])["local_path"] is None
    assert db.get("l1data", bad["id"])["download_status"] == "failed"
    assert download_pending_l1(db, Downloader()) == []


def test_database_download_workers_never_download_same_task_twice(db, tmp_path):
    from collections import Counter
    from threading import Lock
    from roboego_hand_vis.download_data.database_adapter import download_pending_l1
    from roboego_hand_vis.download_data.downloader import DownloadResult

    for i in range(16):
        db.register("l1data", f"ks3://bucket/session{i}")
    calls = []
    lock = Lock()

    class Downloader:
        def download(self, source):
            with lock:
                calls.append(source)
            name = source.rsplit("/", 1)[-1]
            return DownloadResult(tmp_path / name, name)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: download_pending_l1(db, Downloader()), range(4)))
    assert sum(map(len, results)) == 16
    assert len(calls) == 16
    assert all(count == 1 for count in Counter(calls).values())


def test_database_download_limit_leaves_remaining_tasks_pending(db, tmp_path):
    from roboego_hand_vis.download_data.database_adapter import download_pending_l1
    from roboego_hand_vis.download_data.downloader import DownloadResult

    first = db.register("l1data", "ks3://bucket/first")
    second = db.register("l1data", "ks3://bucket/second")

    class Downloader:
        def download(self, source):
            return DownloadResult(tmp_path / "first", "first")

    assert len(download_pending_l1(db, Downloader(), limit=1)) == 1
    assert db.get("l1data", first["id"])["download_status"] == "completed"
    assert db.get("l1data", second["id"])["download_status"] == "pending"


@pytest.mark.parametrize('stage,dependency', [
    ('stereo_depth_gate', 'stereo_rectification'),
    ('fast_foundation_depth', 'stereo_depth_gate'),
])
def test_depth_stages_dependencies_concurrent_claim_and_finish(db, stage, dependency):
    for status in ('pending', 'processing', 'block', 'failed', 'completed'):
        row = db.register('l1data', f'ks3://bucket/{status}')
        db.claim('l1data', row['id'])
        db.complete('l1data', row['id'], status)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET raw_headband_integrity_status='completed', "
                              "video_simple_quality_status='completed', stereo_rectification_status='completed' "
                              "WHERE id=:id"), row)
            conn.execute(text(f'UPDATE l1data SET {dependency}_status=:status WHERE id=:id'),
                         dict(status=status, id=row['id']))
    claim = getattr(db, f'claim_{stage}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: claim(), range(8)))
    winners = [r for r in claims if r is not None]
    assert len(winners) == 1
    winner = winners[0]
    assert winner['local_path'] == 'completed'
    assert db.engine.pool.checkedout() == 0
    finish = getattr(db, f'finish_{stage}')
    outcome = dict(passed=False) if stage == 'stereo_depth_gate' else dict(succeeded=False)
    assert finish(winner['id'], **outcome)
    assert not finish(winner['id'], **outcome)
    assert db.get('l1data', winner['id'])[stage + '_status'] == ('block' if stage == 'stereo_depth_gate' else 'failed')


def test_depth_gate_outcomes_and_depth_blocking(db):
    for index, passed in enumerate((True, False, None)):
        row = db.register('l1data', f'ks3://bucket/{index}')
        db.claim('l1data', row['id'])
        db.complete('l1data', row['id'], str(index))
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET raw_headband_integrity_status='completed', "
                              "video_simple_quality_status='completed', stereo_rectification_status='completed' WHERE id=:id"), row)
        assert db.claim_stereo_depth_gate()['id'] == row['id']
        assert db.finish_stereo_depth_gate(row['id'], passed=passed)
        assert db.get('l1data', row['id'])['stereo_depth_gate_status'] == ('completed' if passed is True else 'block' if passed is False else 'failed')
        claimed = db.claim_fast_foundation_depth()
        if passed is True:
            assert claimed['id'] == row['id']
            assert db.finish_fast_foundation_depth(row['id'], succeeded=True)
        else:
            assert claimed is None


def test_archive_requires_hand_and_depth_completed_and_claims_once(db):
    for dependency in ('fast_foundation_depth_status', 'hand_status'):
        for status in ('pending', 'processing', 'block', 'failed', 'completed'):
            row = db.register('l1data', f'ks3://bucket/{dependency}/{status}')
            db.claim('l1data', row['id'])
            db.complete('l1data', row['id'], f'{dependency}/{status}')
            with db.engine.begin() as conn:
                conn.execute(text("UPDATE l1data SET raw_headband_integrity_status='completed', "
                                  "video_simple_quality_status='completed', stereo_rectification_status='completed', "
                                  "stereo_depth_gate_status='completed', fast_foundation_depth_status='completed', "
                                  "hand_status='completed' WHERE id=:id"), row)
                conn.execute(text(f'UPDATE l1data SET {dependency}=:status WHERE id=:id'), dict(status=status,id=row['id']))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: db.claim_archive_depth_cache(), range(8)))
    winners = [r for r in claims if r is not None]
    assert len(winners) == 2
    assert len({r['id'] for r in winners}) == 2
    assert all(r['local_path'].endswith('/completed') for r in winners)
    assert db.engine.pool.checkedout() == 0
    for row, success in zip(winners, (True, False)):
        assert db.finish_archive_depth_cache(row['id'], succeeded=success)
        assert not db.finish_archive_depth_cache(row['id'], succeeded=success)
        assert db.get('l1data',row['id'])['archive_depth_cache_status'] == ('completed' if success else 'failed')


def test_archive_runner_releases_connection_and_failure_keeps_cache(db, tmp_path):
    from roboego_hand_vis.archive_depth_cache.database_run import run_pending
    from roboego_hand_vis.archive_depth_cache.compute import DepthArchiveConfig
    row = db.register('l1data', 'ks3://bucket/archive')
    db.claim('l1data', row['id'])
    db.complete('l1data', row['id'], 'session')
    with db.engine.begin() as conn:
        conn.execute(text("UPDATE l1data SET raw_headband_integrity_status='completed', "
                          "video_simple_quality_status='completed', stereo_rectification_status='completed', "
                          "stereo_depth_gate_status='completed', fast_foundation_depth_status='completed', "
                          "hand_status='completed' WHERE id=:id"), row)
    depth = tmp_path/'session/depth'
    depth.mkdir(parents=True)
    source = depth/'frame_000000_foundation_depth_m.npy'
    source.write_bytes(b'fake depth')
    def encode(task, config):
        assert db.engine.pool.checkedout() == 0
        assert db.get('l1data',row['id'])['archive_depth_cache_status'] == 'processing'
        raise RuntimeError('simulated encoder error')
    result=run_pending(db,tmp_path,DepthArchiveConfig(),runner=encode)
    assert result[0]['status']=='failed'
    assert source.exists()
    assert db.get('l1data',row['id'])['archive_depth_cache_status']=='failed'


def test_privacy_requires_rectification_and_concurrent_claim_is_unique(db):
    for status in ('pending','processing','block','failed','completed'):
        row=db.register('l1data',f'ks3://bucket/privacy/{status}')
        db.claim('l1data',row['id'])
        db.complete('l1data',row['id'],status)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET raw_headband_integrity_status='completed', "
                              "video_simple_quality_status='completed', stereo_rectification_status=:status WHERE id=:id"),
                         dict(status=status,id=row['id']))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims=list(pool.map(lambda _:db.claim_video_privacy_blur(),range(8)))
    winners=[r for r in claims if r is not None]
    assert len(winners)==1
    row=winners[0]
    assert row['local_path']=='completed'
    assert db.engine.pool.checkedout()==0
    assert db.finish_video_privacy_blur(row['id'],succeeded=True)
    assert not db.finish_video_privacy_blur(row['id'],succeeded=False)
    final=db.get('l1data',row['id'])
    assert final['video_privacy_blur_status']=='completed'
    assert final['fast_foundation_depth_status']=='pending'
    assert final['hand_status']=='pending'


def test_privacy_runner_releases_connection_and_writes_failed(db,tmp_path):
    from roboego_hand_vis.video_privacy_blur.database_run import run_pending
    row=db.register('l1data','ks3://bucket/privacy-error')
    db.claim('l1data',row['id'])
    db.complete('l1data',row['id'],'session')
    with db.engine.begin() as conn:
        conn.execute(text("UPDATE l1data SET raw_headband_integrity_status='completed', "
                          "video_simple_quality_status='completed', stereo_rectification_status='completed' WHERE id=:id"),row)
    videos=tmp_path/'session/videos'
    videos.mkdir(parents=True)
    (videos/'left.mkv').write_bytes(b'video')
    weights=tmp_path/'face.pt'
    weights.write_bytes(b'model')
    def process(config):
        assert db.engine.pool.checkedout()==0
        assert db.get('l1data',row['id'])['video_privacy_blur_status']=='processing'
        raise RuntimeError('simulated detection failure')
    result=run_pending(db,tmp_path,{'paths':{'face_weights':str(weights)}},runner=process)
    assert result[0]['status']=='failed'
    assert db.get('l1data',row['id'])['video_privacy_blur_status']=='failed'


def test_hand_waits_for_depth_and_concurrent_claim_is_unique(db):
    for status in ('pending','processing','block','failed','completed'):
        row=db.register('l1data',f'ks3://bucket/hand/{status}')
        db.claim('l1data',row['id'])
        db.complete('l1data',row['id'],status)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET raw_headband_integrity_status='completed', "
                              "video_simple_quality_status='completed', stereo_rectification_status='completed', "
                              "stereo_depth_gate_status='completed', fast_foundation_depth_status=:status WHERE id=:id"),
                         dict(status=status,id=row['id']))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims=list(pool.map(lambda _:db.claim_hand(),range(8)))
    winners=[r for r in claims if r is not None]
    assert len(winners)==1
    row=winners[0]
    assert row['local_path']=='completed'
    assert row['hand_status']=='processing'
    assert db.engine.pool.checkedout()==0
    assert db.finish_hand(row['id'],succeeded=False)
    assert not db.finish_hand(row['id'],succeeded=True)
    assert db.get('l1data',row['id'])['hand_status']=='failed'
    assert db.claim_archive_depth_cache() is None


def test_hand_success_unlocks_archive(db):
    row=db.register('l1data','ks3://bucket/hand-ready')
    db.claim('l1data',row['id'])
    db.complete('l1data',row['id'],'session')
    with db.engine.begin() as conn:
        conn.execute(text("UPDATE l1data SET raw_headband_integrity_status='completed', "
                          "video_simple_quality_status='completed', stereo_rectification_status='completed', "
                          "stereo_depth_gate_status='completed', fast_foundation_depth_status='completed' WHERE id=:id"),row)
    assert db.claim_archive_depth_cache() is None
    assert db.claim_hand()['id']==row['id']
    assert db.finish_hand(row['id'],succeeded=True)
    assert db.claim_archive_depth_cache()['id']==row['id']


def test_vio_waits_for_rectification_and_claims_once_without_depth(db):
    for status in ('pending','processing','block','failed','completed'):
        row=db.register('l1data',f'ks3://bucket/vio/{status}')
        db.claim('l1data',row['id'])
        db.complete('l1data',row['id'],status)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE l1data SET raw_headband_integrity_status='completed', "
                              "video_simple_quality_status='completed',stereo_rectification_status=:status WHERE id=:id"),
                         dict(status=status,id=row['id']))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims=list(pool.map(lambda _:db.claim_vio(),range(8)))
    winners=[r for r in claims if r is not None]
    assert len(winners)==1
    row=winners[0]
    assert row['local_path']=='completed'
    assert row['vio_status']=='processing'
    assert db.engine.pool.checkedout()==0
    assert row['fast_foundation_depth_status']=='pending'
    assert row['hand_status']=='pending'
    assert db.finish_vio(row['id'],succeeded=True)
    assert not db.finish_vio(row['id'],succeeded=False)
    assert db.get('l1data',row['id'])['vio_status']=='completed'
