"""Deterministic message-ID boundaries for interaction-mode switches.

This module computes the immutable boundary fields stored on
``mode_switch_events``.  It performs no LLM calls and does not write to
the database.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.interaction_modes import CORRECTIVE_MODE, RECEIVE_TEACHING_MODE
from backend.models import Message, ModeSwitchEvent

HISTORY_BOUNDARY_VERSION = "history-boundary-v1"


@dataclass(frozen=True)
class HistoryBoundaryCapture:
    """Result of capturing a boundary for a real mode switch."""

    history_through_message_id: int | None
    history_boundary_version: str
    reviewable_user_message_count: int | None
    lower_bound_message_id: int | None = None
    boundary_reliable: bool = True


def capture_history_boundary(
    *,
    db: Session,
    session_id: int,
    from_mode: str,
    to_mode: str,
) -> HistoryBoundaryCapture:
    """Capture the message-ID boundary for a real mode switch.

    The caller must hold the per-session lock and must not yet have
    flushed the new mode-switch event.
    """
    upper_bound = db.scalar(
        select(func.max(Message.id)).where(
            Message.session_id == session_id,
        )
    )

    if (
        from_mode != RECEIVE_TEACHING_MODE
        or to_mode != CORRECTIVE_MODE
    ):
        return HistoryBoundaryCapture(
            history_through_message_id=upper_bound,
            history_boundary_version=HISTORY_BOUNDARY_VERSION,
            reviewable_user_message_count=None,
        )

    previous_resume_event = db.execute(
        select(ModeSwitchEvent)
        .where(
            ModeSwitchEvent.session_id == session_id,
            ModeSwitchEvent.from_mode == CORRECTIVE_MODE,
            ModeSwitchEvent.to_mode == RECEIVE_TEACHING_MODE,
        )
        .order_by(ModeSwitchEvent.id.desc())
        .limit(1)
    ).scalars().first()

    if previous_resume_event is None:
        lower_bound = None
    elif previous_resume_event.history_boundary_version != HISTORY_BOUNDARY_VERSION:
        return HistoryBoundaryCapture(
            history_through_message_id=upper_bound,
            history_boundary_version=HISTORY_BOUNDARY_VERSION,
            reviewable_user_message_count=None,
            lower_bound_message_id=None,
            boundary_reliable=False,
        )
    else:
        lower_bound = previous_resume_event.history_through_message_id

    if upper_bound is None:
        return HistoryBoundaryCapture(
            history_through_message_id=None,
            history_boundary_version=HISTORY_BOUNDARY_VERSION,
            reviewable_user_message_count=0,
            lower_bound_message_id=lower_bound,
            boundary_reliable=True,
        )

    count_stmt = select(func.count()).select_from(Message).where(
        Message.session_id == session_id,
        Message.role == "user",
        Message.interaction_mode_snapshot == RECEIVE_TEACHING_MODE,
        Message.id <= upper_bound,
    )
    if lower_bound is not None:
        count_stmt = count_stmt.where(Message.id > lower_bound)

    count = db.execute(count_stmt).scalar()
    return HistoryBoundaryCapture(
        history_through_message_id=upper_bound,
        history_boundary_version=HISTORY_BOUNDARY_VERSION,
        reviewable_user_message_count=int(count or 0),
        lower_bound_message_id=lower_bound,
        boundary_reliable=True,
    )
