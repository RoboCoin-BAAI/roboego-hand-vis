"""Transactional registration and download task operations."""

from sqlalchemy import MetaData, Table, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from .database import Database
from .model import TaskStatus, get_table
from .stage_schema import STAGE_REQUIRED_COLUMNS
from .validators import validate_ks3_path, validate_relative_path


class DataService:
    """Thread-safe operations; construct a new engine in each worker process.

    Each method commits before returning. A successful claim is the only
    authorization to start a download. There is no automatic recovery of active
    claims because the four-column schema has no ownership or lease fields.
    """

    def __init__(self, engine: Engine | Database, *, table_name: str = "l1data"):
        self.table_name = get_table(table_name).name
        self.database = engine if isinstance(engine, Database) else Database(engine)
        self._reflected_stage_table = None

    @property
    def engine(self) -> Engine:
        """Expose the underlying engine for compatibility with existing callers."""
        return self.database.engine

    @classmethod
    def from_url(cls, url: str, *, table_name: str = "l1data") -> "DataService":
        """Create a service using a postgresql+psycopg:// connection URL."""
        get_table(table_name)
        return cls(Database.from_url(url), table_name=table_name)

    def _stage_table(self, stage: str) -> Table:
        """Validate real columns before a stage can claim or update any row."""
        required = STAGE_REQUIRED_COLUMNS[stage]
        if self._reflected_stage_table is None:
            self._reflected_stage_table = Table(
                self.table_name, MetaData(), autoload_with=self.engine
            )
        table = self._reflected_stage_table
        missing = sorted(set(required) - set(table.c.keys()))
        if missing:
            raise ValueError(
                f"表 {self.table_name} 无法运行 {stage}，缺少字段：{', '.join(missing)}；"
                "请先更新目标表结构。"
            )
        return table

    def validate_stage(self, stage: str) -> None:
        """Preflight only; does not claim tasks or change schema."""
        self._stage_table(stage)

    def initialize(self) -> None:
        """Delegate explicit schema initialization to the connection layer."""
        self.database.initialize()

    def register(self, table_name: str, ks3_path: str) -> dict:
        """Register a KS3 path, returning its existing row on a duplicate."""
        row, _ = self.register_if_absent(table_name, ks3_path)
        return row

    def register_if_absent(self, table_name: str, ks3_path: str) -> tuple[dict, bool]:
        """Return (record, created); duplicates preserve every existing field.

        The UNIQUE constraint decides which concurrent caller creates the row.
        Use READ COMMITTED isolation, as provided by Database.from_url().
        """
        table = get_table(table_name)
        validate_ks3_path(ks3_path)
        stmt = insert(table).values(ks3_path=ks3_path)
        stmt = stmt.on_conflict_do_nothing(index_elements=[table.c.ks3_path]).returning(table)
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            if row is not None:
                return dict(row), True
            # A new READ COMMITTED snapshot sees the winning insert after the
            # conflict check waited for its transaction to finish.
            row = session.execute(
                select(table).where(table.c.ks3_path == ks3_path)
            ).mappings().one()
            return dict(row), False

    def get(self, table_name: str, data_id: int) -> dict | None:
        table = get_table(table_name)
        with self.database.with_session() as session:
            row = session.execute(select(table).where(table.c.id == data_id)).mappings().first()
            return dict(row) if row is not None else None

    def claim(self, table_name: str, data_id: int | None = None) -> dict | None:
        """Atomically claim a pending ID, or the next unlocked pending record.

        None means no task was claimed, possibly because another transaction
        currently holds the row lock. Callers may poll again later.
        """
        table = get_table(table_name)
        candidate = select(table.c.id).where(table.c.download_status == TaskStatus.PENDING.value)
        if data_id is not None:
            candidate = candidate.where(table.c.id == data_id)
        candidate = candidate.order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte()
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(download_status=TaskStatus.PROCESSING.value, local_path=None).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def complete(self, table_name: str, data_id: int, local_path: str) -> bool:
        """Mark complete AFTER the claim owner publishes the shared file."""
        return self._transition(table_name, data_id, TaskStatus.PROCESSING.value, TaskStatus.COMPLETED.value,
                                validate_relative_path(local_path))

    def claim_raw_headband_integrity(self) -> dict | None:
        """Claim an already downloaded L1 session for its first quality gate."""
        table = self._stage_table('raw_headband_integrity')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None),
            table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(raw_headband_integrity_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_raw_headband_integrity(self, data_id: int, *, passed: bool | None) -> bool:
        """True completes, False blocks, None records execution failure."""
        if passed is not True and passed is not False and passed is not None:
            raise ValueError("passed must be True, False or None")
        table = self._stage_table('raw_headband_integrity')
        target = (TaskStatus.COMPLETED.value if passed is True else
                  TaskStatus.BLOCK.value if passed is False else TaskStatus.FAILED.value)
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.raw_headband_integrity_status == TaskStatus.PROCESSING.value,
        ).values(raw_headband_integrity_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None

    def fail(self, table_name: str, data_id: int) -> bool:
        """The claim owner marks its unsuccessful download failed."""
        return self._transition(table_name, data_id, TaskStatus.PROCESSING.value, TaskStatus.FAILED.value)

    def claim_video_simple_quality(self) -> dict | None:
        """Claim a downloaded L1 session that passed raw integrity checking."""
        table = self._stage_table('video_simple_quality')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None),
            table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(video_simple_quality_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_video_simple_quality(self, data_id: int, *, passed: bool | None) -> bool:
        """True completes, False blocks, None records execution failure."""
        if passed is not True and passed is not False and passed is not None:
            raise ValueError("passed must be True, False or None")
        table = self._stage_table('video_simple_quality')
        target = (TaskStatus.COMPLETED.value if passed is True else
                  TaskStatus.BLOCK.value if passed is False else TaskStatus.FAILED.value)
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.video_simple_quality_status == TaskStatus.PROCESSING.value,
        ).values(video_simple_quality_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None

    def retry(self, table_name: str, data_id: int) -> bool:
        """Explicitly return a failed task to the queue; active tasks are untouched."""
        return self._transition(table_name, data_id, TaskStatus.FAILED.value, TaskStatus.PENDING.value)

    def claim_detect_hand(self) -> dict | None:
        """Claim optional hand detection after both preceding quality gates."""
        table = self._stage_table('detect_hand')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None),
            table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.detect_hand_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(detect_hand_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_detect_hand(self, data_id: int, *, succeeded: bool) -> bool:
        """Record execution outcome only; hand presence is not a quality gate."""
        if succeeded is not True and succeeded is not False:
            raise ValueError("succeeded must be True or False")
        table = self._stage_table('detect_hand')
        target = TaskStatus.COMPLETED.value if succeeded else TaskStatus.FAILED.value
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.detect_hand_status == TaskStatus.PROCESSING.value,
        ).values(detect_hand_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None

    def _transition(self, table_name: str, data_id: int, expected: str,
                    target: str, local_path: str | None = None) -> bool:
        table = get_table(table_name)
        stmt = (update(table).where(table.c.id == data_id, table.c.download_status == expected)
                .values(download_status=target, local_path=local_path).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None

    def claim_stereo_rectification(self) -> dict | None:
        """Claim rectification after the two required gates; hand is optional."""
        table = self._stage_table('stereo_rectification')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None), table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.stereo_rectification_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(stereo_rectification_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_stereo_rectification(self, data_id: int, *, succeeded: bool) -> bool:
        """Record processing outcome; rectification is not a quality gate."""
        if succeeded is not True and succeeded is not False:
            raise ValueError("succeeded must be True or False")
        table = self._stage_table('stereo_rectification')
        target = TaskStatus.COMPLETED.value if succeeded else TaskStatus.FAILED.value
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.stereo_rectification_status == TaskStatus.PROCESSING.value,
        ).values(stereo_rectification_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None

    def claim_qwen_annotation(self) -> dict | None:
        """Claim original-left annotation after the two required quality gates."""
        table = self._stage_table('qwen_annotation')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None), table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.qwen_annotation_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(qwen_annotation_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_qwen_annotation(self, data_id: int, *, succeeded: bool) -> bool:
        """Record annotation completeness; semantic contents are not a gate."""
        if succeeded is not True and succeeded is not False:
            raise ValueError("succeeded must be True or False")
        table = self._stage_table('qwen_annotation')
        target = TaskStatus.COMPLETED.value if succeeded else TaskStatus.FAILED.value
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.qwen_annotation_status == TaskStatus.PROCESSING.value,
        ).values(qwen_annotation_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None

    def claim_post_annotation_actions(self) -> dict | None:
        """Claim action standardization only after complete Qwen annotation."""
        table = self._stage_table('post_annotation_actions')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None), table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.qwen_annotation_status == TaskStatus.COMPLETED.value,
            table.c.post_annotation_actions_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(post_annotation_actions_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_post_annotation_actions(self, data_id: int, *, succeeded: bool) -> bool:
        """Record standardization outcome; this stage is not a gate."""
        if succeeded is not True and succeeded is not False:
            raise ValueError("succeeded must be True or False")
        table = self._stage_table('post_annotation_actions')
        target = TaskStatus.COMPLETED.value if succeeded else TaskStatus.FAILED.value
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.post_annotation_actions_status == TaskStatus.PROCESSING.value,
        ).values(post_annotation_actions_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None


    def claim_stereo_depth_gate(self) -> dict | None:
        """Claim a downloaded session after required quality gates and stereo_rectification."""
        table = self._stage_table('stereo_depth_gate')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None), table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.stereo_rectification_status == TaskStatus.COMPLETED.value,
            table.c.stereo_depth_gate_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(stereo_depth_gate_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_stereo_depth_gate(self, data_id: int, *, passed: bool | None) -> bool:
        """Record this stage's outcome only while it is processing."""
        if passed is not True and passed is not False and passed is not None:
            raise ValueError("passed must be True, False or None")
        table = self._stage_table('stereo_depth_gate')
        target = (TaskStatus.COMPLETED.value if passed is True else
                  TaskStatus.BLOCK.value if passed is False else TaskStatus.FAILED.value)
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.stereo_depth_gate_status == TaskStatus.PROCESSING.value,
        ).values(stereo_depth_gate_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None


    def claim_fast_foundation_depth(self) -> dict | None:
        """Claim a rectified session after basic quality checks, independently of the depth gate."""
        table = self._stage_table('fast_foundation_depth')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None), table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.stereo_rectification_status == TaskStatus.COMPLETED.value,
            table.c.fast_foundation_depth_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(fast_foundation_depth_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_fast_foundation_depth(self, data_id: int, *, succeeded: bool) -> bool:
        """Record this stage's outcome only while it is processing."""
        if succeeded is not True and succeeded is not False:
            raise ValueError("succeeded must be True or False")
        table = self._stage_table('fast_foundation_depth')
        target = TaskStatus.COMPLETED.value if succeeded else TaskStatus.FAILED.value
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.fast_foundation_depth_status == TaskStatus.PROCESSING.value,
        ).values(fast_foundation_depth_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None


    def claim_archive_depth_cache(self) -> dict | None:
        """Claim archive only after depth and hand estimation have completed."""
        table = self._stage_table('archive_depth_cache')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None), table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.stereo_rectification_status == TaskStatus.COMPLETED.value,
            table.c.stereo_depth_gate_status == TaskStatus.COMPLETED.value,
            table.c.fast_foundation_depth_status == TaskStatus.COMPLETED.value,
            table.c.hand_status == TaskStatus.COMPLETED.value,
            table.c.archive_depth_cache_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(archive_depth_cache_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_archive_depth_cache(self, data_id: int, *, succeeded: bool) -> bool:
        """Record this stage's outcome only while it is processing."""
        if succeeded is not True and succeeded is not False:
            raise ValueError("succeeded must be True or False")
        table = self._stage_table('archive_depth_cache')
        target = TaskStatus.COMPLETED.value if succeeded else TaskStatus.FAILED.value
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.archive_depth_cache_status == TaskStatus.PROCESSING.value,
        ).values(archive_depth_cache_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None


    def claim_video_privacy_blur(self) -> dict | None:
        """Claim privacy blur after rectification and required quality gates."""
        table = self._stage_table('video_privacy_blur')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None), table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.stereo_rectification_status == TaskStatus.COMPLETED.value,
            table.c.video_privacy_blur_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(video_privacy_blur_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_video_privacy_blur(self, data_id: int, *, succeeded: bool) -> bool:
        """Record privacy processing outcome; this stage is not a quality gate."""
        if succeeded is not True and succeeded is not False:
            raise ValueError("succeeded must be True or False")
        table = self._stage_table('video_privacy_blur')
        target = TaskStatus.COMPLETED.value if succeeded else TaskStatus.FAILED.value
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.video_privacy_blur_status == TaskStatus.PROCESSING.value,
        ).values(video_privacy_blur_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None


    def claim_hand(self) -> dict | None:
        """Claim native hand estimation only after depth estimation has completed."""
        table = self._stage_table('hand')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None), table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.stereo_rectification_status == TaskStatus.COMPLETED.value,
            table.c.stereo_depth_gate_status == TaskStatus.COMPLETED.value,
            table.c.fast_foundation_depth_status == TaskStatus.COMPLETED.value,
            table.c.hand_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(hand_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_hand(self, data_id: int, *, succeeded: bool) -> bool:
        """Record this stage's outcome only while it is processing."""
        if succeeded is not True and succeeded is not False:
            raise ValueError("succeeded must be True or False")
        table = self._stage_table('hand')
        target = TaskStatus.COMPLETED.value if succeeded else TaskStatus.FAILED.value
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.hand_status == TaskStatus.PROCESSING.value,
        ).values(hand_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None


    def claim_vio(self) -> dict | None:
        """Claim rectified VIO after rectification and required quality gates."""
        table = self._stage_table('vio')
        candidate = (select(table.c.id).where(
            table.c.download_status == TaskStatus.COMPLETED.value,
            table.c.local_path.is_not(None), table.c.local_path != "",
            table.c.raw_headband_integrity_status == TaskStatus.COMPLETED.value,
            table.c.video_simple_quality_status == TaskStatus.COMPLETED.value,
            table.c.stereo_rectification_status == TaskStatus.COMPLETED.value,
            table.c.vio_status == TaskStatus.PENDING.value,
        ).order_by(table.c.id).limit(1).with_for_update(skip_locked=True).cte())
        stmt = (update(table).where(table.c.id == candidate.c.id)
                .values(vio_status=TaskStatus.PROCESSING.value).returning(table))
        with self.database.with_session() as session:
            row = session.execute(stmt).mappings().first()
            return dict(row) if row is not None else None

    def finish_vio(self, data_id: int, *, succeeded: bool) -> bool:
        """Record VIO execution outcome; this stage is not a quality gate."""
        if succeeded is not True and succeeded is not False:
            raise ValueError("succeeded must be True or False")
        table = self._stage_table('vio')
        target = TaskStatus.COMPLETED.value if succeeded else TaskStatus.FAILED.value
        stmt = (update(table).where(
            table.c.id == data_id,
            table.c.vio_status == TaskStatus.PROCESSING.value,
        ).values(vio_status=target).returning(table.c.id))
        with self.database.with_session() as session:
            return session.execute(stmt).first() is not None
