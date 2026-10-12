"""ORM models and schema definitions; no connections are opened here."""

from enum import Enum

from sqlalchemy import BigInteger, CheckConstraint, Identity, Index, Table, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCK = "block"


class Base(DeclarativeBase):
    pass


def _table_constraints(name: str) -> tuple:
    """Build independent constraint/index objects for each mapped table."""
    status_values = ", ".join(f"'{status.value}'" for status in TaskStatus)
    return (
        CheckConstraint("length(btrim(ks3_path)) > 0", name=f"{name}_ks3_nonempty"),
        CheckConstraint(
            f"download_status IN ({status_values})", name=f"{name}_status_valid"
        ),
        CheckConstraint(
            "download_status <> 'completed' OR "
            "(local_path IS NOT NULL AND length(btrim(local_path)) > 0)",
            name=f"{name}_completed_path",
        ),
        Index(f"{name}_pending_id", "id", postgresql_where=text("download_status = 'pending'")),
    )


L1_STAGE_STATUS_FIELDS = (
    "raw_headband_integrity_status",
    "video_simple_quality_status",
    "detect_hand_status",
    "stereo_rectification_status",
    "qwen_annotation_status",
    "post_annotation_actions_status",
    "stereo_depth_gate_status",
    "fast_foundation_depth_status",
    "hand_status",
    "archive_depth_cache_status",
    "vio_status",
    "video_privacy_blur_status",
)


class L1Data(Base):
    """L1 data. Add L1-specific mapped columns here."""

    __tablename__ = "l1data"
    __table_args__ = _table_constraints(__tablename__) + tuple(
        CheckConstraint(
            f"{field} IN ({', '.join(repr(status.value) for status in TaskStatus)})",
            name=f"l1data_{field}_valid",
        )
        for field in L1_STAGE_STATUS_FIELDS
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    ks3_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Keep the database column TEXT, with a CHECK constraint for allowed values.
    download_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)


    # L1 processing stages; explicit fields correspond to config/l1_fields.yaml.
    raw_headband_integrity_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    video_simple_quality_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    detect_hand_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    stereo_rectification_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    qwen_annotation_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    post_annotation_actions_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    stereo_depth_gate_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    fast_foundation_depth_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    hand_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    archive_depth_cache_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    vio_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    video_privacy_blur_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )

class L2Data(Base):
    """L2 data. Add L2-specific mapped columns here."""

    __tablename__ = "l2data"
    __table_args__ = _table_constraints(__tablename__)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    ks3_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Keep the database column TEXT, with a CHECK constraint for allowed values.
    download_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class L3Data(Base):
    """L3 data. Add L3-specific mapped columns here."""

    __tablename__ = "l3data"
    __table_args__ = _table_constraints(__tablename__)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    ks3_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Keep the database column TEXT, with a CHECK constraint for allowed values.
    download_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=TaskStatus.PENDING.value
    )
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)


MODELS = {model.__tablename__: model for model in (L1Data, L2Data, L3Data)}


def get_table(name: str) -> Table:
    """Resolve only known model tables; accept historical mixed-case names."""
    try:
        return MODELS[name.lower()].__table__
    except KeyError:
        raise ValueError("table must be l1data, l2data, or l3data") from None
