"""SQLAlchemy model for investigation history."""

import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Investigation(Base):
    __tablename__ = "investigations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    n_rows: Mapped[int] = mapped_column(Integer)
    n_cols: Mapped[int] = mapped_column(Integer)
    target_column: Mapped[str] = mapped_column(String(255), nullable=True)
    task_type: Mapped[str] = mapped_column(String(50))
    health_score: Mapped[float] = mapped_column(Float)
    grade: Mapped[str] = mapped_column(String(5))
    findings_count: Mapped[int] = mapped_column(Integer)
    report_json: Mapped[dict] = mapped_column(JSON)  # stores detective findings + report text