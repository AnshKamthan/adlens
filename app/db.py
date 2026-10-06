"""Persistence with SQLAlchemy 2.0 (sync).

Production notes:
  * Schema changes go through migrations (Alembic), never `create_all` in prod.
  * Postgres connections are expensive; the engine keeps a pool. Size it per replica:
    replicas x (pool_size + max_overflow) must stay under Postgres max_connections.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Engine,
    Float,
    Index,
    String,
    Text,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from app.schemas import Classification, StoredClassification


class Base(DeclarativeBase):
    pass


class ClassificationRecord(Base):
    __tablename__ = "classifications"
    # "latest result for this ad" = WHERE ad_id = ? ORDER BY created_at DESC LIMIT 1.
    # A composite index on (ad_id, created_at) serves that query without a sort.
    __table_args__ = (Index("ix_classifications_ad_created", "ad_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ad_id: Mapped[str] = mapped_column(String(64))
    brand: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def to_schema(self) -> StoredClassification:
        return StoredClassification(
            ad_id=self.ad_id,
            brand=self.brand,
            category=self.category,
            confidence=self.confidence,
            rationale=self.rationale,
            model=self.model,
            prompt_version=self.prompt_version,
        )


def make_engine(url: str) -> Engine:
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)  # use psycopg 3
    if url.startswith("sqlite"):
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool  # one shared in-memory DB across threads (tests)
        return create_engine(url, **kwargs)
    return create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def save_classification(
    sessions: sessionmaker, ad_id: str, result: Classification, model: str, prompt_version: str
) -> None:
    with sessions.begin() as s:  # commits on success, rolls back on exception
        s.add(
            ClassificationRecord(
                ad_id=ad_id,
                brand=result.brand,
                category=result.category,
                confidence=result.confidence,
                rationale=result.rationale,
                model=model,
                prompt_version=prompt_version,
            )
        )


def get_latest(sessions: sessionmaker, ad_id: str) -> StoredClassification | None:
    stmt = (
        select(ClassificationRecord)
        .where(ClassificationRecord.ad_id == ad_id)  # parameterised: no SQL injection
        .order_by(ClassificationRecord.created_at.desc(), ClassificationRecord.id.desc())
        .limit(1)
    )
    with sessions() as s:
        row = s.scalars(stmt).first()
        return row.to_schema() if row else None


def list_recent(
    sessions: sessionmaker, category: str | None = None, limit: int = 10
) -> list[StoredClassification]:
    stmt = select(ClassificationRecord).order_by(ClassificationRecord.id.desc()).limit(limit)
    if category:
        stmt = stmt.where(ClassificationRecord.category == category)
    with sessions() as s:
        return [r.to_schema() for r in s.scalars(stmt)]


def db_ping(engine: Engine) -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
