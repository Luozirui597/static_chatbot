"""SQLAlchemy ORM models — ChatSession and Message."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import List

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


def utc_now() -> datetime:
    """Return a naive UTC datetime for SQLite storage.

    ``datetime.utcnow`` is deprecated as of Python 3.12.  Use
    ``datetime.now(UTC).replace(tzinfo=None)`` instead so SQLite (which
    stores datetimes as strings) works reliably.
    """
    return datetime.now(UTC).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ChatSession
# ---------------------------------------------------------------------------


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    title: Mapped[str] = mapped_column(
        String(255), default="New Chat", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=utc_now, onupdate=utc_now, nullable=False
    )

    title_is_manual: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="0",
        nullable=False,
    )

    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} title={self.title!r}>"


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        # Only "user" and "assistant" roles are valid.
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_messages_role",
        ),
        # Reject purely-whitespace content: empty string, spaces, tabs,
        # newlines, carriage returns, and any mix of them.
        # trim(X, Y) removes characters in Y from both ends of X.
        # char(9)=TAB, char(10)=LF, char(13)=CR, char(32)=SPACE
        CheckConstraint(
            (
                "length(trim(content, "
                "char(9) || char(10) || char(13) || char(32)"
                ")) > 0"
            ),
            name="ck_messages_content_not_blank",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=utc_now, nullable=False
    )

    session: Mapped["ChatSession"] = relationship(
        "ChatSession", back_populates="messages"
    )

    def __repr__(self) -> str:
        return (
            f"<Message id={self.id} session_id={self.session_id}"
            f" role={self.role!r}>"
        )


# ---------------------------------------------------------------------------
# SchemaMigration — migration bookkeeping
# ---------------------------------------------------------------------------


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version: Mapped[str] = mapped_column(
        String(255), primary_key=True
    )
    applied_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
