"""SQLAlchemy models for ChainInsight."""

import threading
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class PipelineRun(Base):
    """Tracks each pipeline execution."""
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), unique=True, nullable=False)
    status = Column(String(20), default="pending")
    source_file = Column(String(256))
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    etl_stats = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    snapshots = relationship("InventorySnapshot", back_populates="pipeline_run", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="pipeline_run", cascade="all, delete-orphan")


class InventorySnapshot(Base):
    """One row per product per ingestion batch."""
    __tablename__ = "inventory_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), ForeignKey("pipeline_runs.batch_id"), index=True, nullable=False)
    ingested_at = Column(DateTime, index=True, default=lambda: datetime.now(timezone.utc))
    product_id = Column(String(32), index=True)
    category = Column(String(32))
    unit_cost = Column(Float)
    current_stock = Column(Float)
    daily_demand_est = Column(Float)
    safety_stock_target = Column(Float)
    vendor_name = Column(String(64))
    lead_time_days = Column(Float)
    reorder_point = Column(Float)
    stock_status = Column(String(20))
    inventory_value = Column(Float)

    pipeline_run = relationship("PipelineRun", back_populates="snapshots")


class AnalysisResult(Base):
    """Stores KPI / chart metadata per pipeline run."""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), ForeignKey("pipeline_runs.batch_id"), index=True, nullable=False)
    analysis_type = Column(String(32), nullable=False)
    result_json = Column(JSON)
    chart_paths = Column(JSON)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    pipeline_run = relationship("PipelineRun", back_populates="analysis_results")


# ---- Engine & Session factory (lazy) ----

_engine = None


def get_engine():
    """Return the shared SQLAlchemy engine, creating it lazily on first call."""
    global _engine
    if _engine is None:
        from app.config import DATABASE_URL
        connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
        _engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)
    return _engine


def get_session_local():
    """Return a new sessionmaker bound to the current engine."""
    return sessionmaker(bind=get_engine())


def init_db():
    """Create all tables."""
    Base.metadata.create_all(get_engine())


# Module-level convenience alias (lazy — evaluates engine on first use of the factory).
class _LazySessionLocal:
    """Proxy so that ``SessionLocal()`` works without eagerly creating the engine."""

    _factory = None
    _lock = threading.Lock()

    def __call__(self, **kw):
        with self._lock:
            if self._factory is None:
                self._factory = get_session_local()
        return self._factory(**kw)

    def configure(self, **kw):
        if self._factory is None:
            self._factory = get_session_local()
        self._factory.configure(**kw)


SessionLocal = _LazySessionLocal()


def get_db():
    """Yield a DB session (for FastAPI dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
