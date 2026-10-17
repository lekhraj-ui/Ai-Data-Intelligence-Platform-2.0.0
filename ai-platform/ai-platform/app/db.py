import logging
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class EventLog(Base):
    __tablename__ = "event_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(50), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


def init_db() -> None:
    Base.metadata.create_all(engine)


def log_event(kind: str, payload: dict, result: dict, latency_ms: float = 0.0) -> None:
    """Persist an audit record. Never lets a DB problem break an API call."""
    try:
        with SessionLocal() as s:
            s.add(EventLog(kind=kind, payload=payload, result=result, latency_ms=latency_ms))
            s.commit()
    except Exception:  # noqa: BLE001
        logger.exception("could not log event")


def recent_events(limit: int = 20) -> list[dict]:
    with SessionLocal() as s:
        rows = s.scalars(select(EventLog).order_by(EventLog.id.desc()).limit(limit)).all()
        return [
            {"id": r.id, "kind": r.kind, "latency_ms": r.latency_ms,
             "created_at": r.created_at.isoformat(), "result": r.result}
            for r in rows
        ]
