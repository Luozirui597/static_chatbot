"""SQLAlchemy models for the isolated strategy-stability integration.

These tables are intentionally local to the experiment.  They are never
registered on, or migrated into, the production database.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentBase(DeclarativeBase):
    pass


class ExperimentSession(ExperimentBase):
    __tablename__ = "experiment_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(16), nullable=False)
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en")
    contract_version: Mapped[str] = mapped_column(String(64), nullable=False)
    verifier_version: Mapped[str] = mapped_column(String(64), nullable=False)
    behavior_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    turn_context_json: Mapped[str] = mapped_column(Text, nullable=False)
    asked: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    remediated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    closed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pending_gap_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    messages: Mapped[list["ExperimentMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ExperimentMessage(ExperimentBase):
    __tablename__ = "experiment_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_sessions.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    strategy_snapshot: Mapped[str] = mapped_column(String(16), nullable=False)
    language_snapshot: Mapped[str] = mapped_column(String(8), nullable=False)
    model_snapshot: Mapped[str] = mapped_column(String(128), nullable=False)
    contract_version_snapshot: Mapped[str] = mapped_column(String(64), nullable=False)
    verifier_version_snapshot: Mapped[str] = mapped_column(String(64), nullable=False)
    behavior_version_snapshot: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    session: Mapped[ExperimentSession] = relationship(back_populates="messages")
