"""PostgreSQL connection pool, session lifecycle, and schema initialization."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .model import Base


class Database:
    """Share the factory across threads, but create it separately per process."""

    def __init__(self, engine: Engine):
        if engine.dialect.name != "postgresql":
            raise ValueError("Database requires PostgreSQL")
        self.engine = engine
        self.session_factory = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )

    @classmethod
    def from_url(cls, url: str) -> "Database":
        """Create a pool without connecting or creating any tables."""
        return cls(create_engine(url, pool_pre_ping=True, isolation_level="READ COMMITTED"))

    @contextmanager
    def with_session(self) -> Iterator[Session]:
        """Commit on success, roll back on failure, and always close the session."""
        with self.session_factory.begin() as session:
            yield session

    def initialize(self) -> None:
        """Create missing tables; changing existing columns requires a migration."""
        with self.engine.begin() as conn:
            # Serialize initializers, including their table-existence checks.
            conn.execute(text("SELECT pg_advisory_xact_lock(7210060901)"))
            Base.metadata.create_all(conn)

    def close(self) -> None:
        """Dispose the connection pool after all workers using it have stopped."""
        self.engine.dispose()
