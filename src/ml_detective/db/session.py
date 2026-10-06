"""Engine/session setup, reading DATABASE_URL from settings (Phase 2)."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ml_detective.config.settings import settings
from ml_detective.db.models import Base

_engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {})
_SessionLocal = sessionmaker(bind=_engine)


def init_db() -> None:
    """Creates tables if they don't exist. Safe to call repeatedly."""
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    return _SessionLocal()