"""Window boundary and deterministic programmatic retrieval decisions.

This module is experiment-only.  It does not modify the production prompt or
ChatService.  It deliberately supports a small, explicit English subset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.models import Message

Action = Literal["direct", "retrieve", "clarify", "boundary_unknown"]
Intent = Literal[
    "ordinary_teaching",
    "simple_confirmation",
    "current_fact",
    "history_compare",
    "unsupported",
    "ambiguous_reference",
]
Polarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class WindowBoundary:
    known: bool
    session_id: int
    current_message_id: int
    recent: tuple[Message, ...] = ()
    visible_teacher: tuple[Message, ...] = ()
    visible_start_id: int | None = None
    count_before: int = 0
    window_overflow: bool = False
    hidden_teacher_exists: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ParsedStatement:
    message_id: int
    subject: str
    relation: str
    object: str
    polarity: Polarity
    kind: str
    content: str


@dataclass(frozen=True)
class ParsedQuestion:
    subject: str
    relation: str
    object: str
    intent: Intent
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    query: str | None = None
    question: ParsedQuestion | None = None
    visible_effective: ParsedStatement | None = None


_ENTITY = r"[a-z0-9_]+"
_RELATION = r"links?"
_FACT_RE = re.compile(
    rf"^(?:fact:\s*|i teach you this fact:\s*)?"
    rf"(?P<subject>{_ENTITY})\s+(?P<relation>{_RELATION})\s+"
    rf"(?:to\s+)?(?P<object>{_ENTITY})\.?$",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    rf"^correction:\s*(?P<subject>{_ENTITY})\s+"
    rf"(?P<negative>does\s+not\s+)?(?P<relation>{_RELATION})\s+"
    rf"(?:to\s+)?(?P<object>{_ENTITY})\.?$",
    re.IGNORECASE,
)
_DOES_RE = re.compile(
    rf"^\s*does\s+(?P<subject>{_ENTITY})\s+(?P<relation>{_RELATION})\s+"
    rf"(?:to\s+)?(?P<object>{_ENTITY})\?\s*$",
    re.IGNORECASE,
)
_MIXED_RE = re.compile(
    rf"^\s*i\s+teach\s+you\s+this\s+fact:\s*"
    rf"{_ENTITY}\s+{_RELATION}\s+(?:to\s+)?{_ENTITY}\.\s*"
    rf"(?P<question>does\s+{_ENTITY}\s+{_RELATION}\s+"
    rf"(?:to\s+)?{_ENTITY}\?)\s*$",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r'^\s*compare\s+the\s+original\s+and\s+current\s+version\s+of\s+'
    r'["“](?P<relation_text>.+?)["”]\s*\.?\s*$',
    re.IGNORECASE,
)
_QUOTED_RELATION_RE = re.compile(
    rf"(?P<subject>{_ENTITY})\s+(?P<relation>{_RELATION})\s+"
    rf"(?:to\s+)?(?P<object>{_ENTITY})\.?",
    re.IGNORECASE,
)
_HISTORY_CUE_RE = re.compile(
    r"\b(original|initially|first|ever|before|after|correct|correction|"
    r"corrected|revoke|revoked|withdraw|withdrawn|source|compare)\b",
    re.IGNORECASE,
)
_TEACHING_RE = re.compile(
    rf"^\s*i\s+teach\s+you\s+this\s+fact:\s*"
    rf"{_ENTITY}\s+{_RELATION}\s+(?:to\s+)?{_ENTITY}\.\s*$",
    re.IGNORECASE,
)
_SIMPLE_CONFIRM_RE = re.compile(
    r"^\s*(?:ok|okay|understood|got\s+it|thanks)[.!]?\s*$",
    re.IGNORECASE,
)


def load_window_boundary(
    db: Session,
    *,
    session_id: int,
    current_message_id: int,
) -> WindowBoundary:
    current = db.get(Message, current_message_id)
    if current is None or current.session_id != session_id or current.role != "user":
        return WindowBoundary(
            known=False,
            session_id=session_id,
            current_message_id=current_message_id,
            error="current_message_invalid",
        )
    rows = (
        db.execute(
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.id < current_message_id,
            )
            .order_by(Message.id.desc())
            .limit(20)
        )
        .scalars()
        .all()
    )
    recent = tuple(reversed(rows))
    count_before = int(
        db.scalar(
            select(func.count(Message.id)).where(
                Message.session_id == session_id,
                Message.id < current_message_id,
            )
        )
        or 0
    )
    visible_teacher = tuple(m for m in recent if m.role == "user")
    visible_start_id = min((m.id for m in recent), default=None)
    hidden_teacher_exists = False
    if visible_start_id is not None:
        hidden_teacher_exists = (
            db.execute(
                select(Message.id)
                .where(
                    Message.session_id == session_id,
                    Message.role == "user",
                    Message.id < visible_start_id,
                )
                .limit(1)
            ).first()
            is not None
        )
    return WindowBoundary(
        known=True,
        session_id=session_id,
        current_message_id=current_message_id,
        recent=recent,
        visible_teacher=visible_teacher,
        visible_start_id=visible_start_id,
        count_before=count_before,
        window_overflow=count_before > 20,
        hidden_teacher_exists=hidden_teacher_exists,
    )


def parse_statement(message: Message) -> ParsedStatement | None:
    text = " ".join(str(message.content).split())
    match = _FACT_RE.fullmatch(text)
    if match:
        return ParsedStatement(
            message_id=message.id,
            subject=match.group("subject").lower(),
            relation="links",
            object=match.group("object").lower(),
            polarity="positive",
            kind="fact",
            content=message.content,
        )
    match = _CORRECTION_RE.fullmatch(text)
    if match:
        return ParsedStatement(
            message_id=message.id,
            subject=match.group("subject").lower(),
            relation="links",
            object=match.group("object").lower(),
            polarity="negative" if match.group("negative") else "positive",
            kind="correction",
            content=message.content,
        )
    return None


def _parse_full_question(text: str) -> ParsedQuestion | None:
    compare = _COMPARE_RE.fullmatch(text)
    if compare:
        relation = _QUOTED_RELATION_RE.fullmatch(
            " ".join(compare.group("relation_text").split())
        )
        if relation:
            return ParsedQuestion(
                subject=relation.group("subject").lower(),
                relation="links",
                object=relation.group("object").lower(),
                intent="history_compare",
            )
        return ParsedQuestion("", "", "", "unsupported", "compare_relation_unparsed")
    does = _DOES_RE.fullmatch(text)
    if does:
        return ParsedQuestion(
            subject=does.group("subject").lower(),
            relation="links",
            object=does.group("object").lower(),
            intent="current_fact",
        )
    return None


def _parse_question(text: str) -> ParsedQuestion | None:
    normalized = " ".join(text.split())
    parsed = _parse_full_question(normalized)
    if parsed is not None:
        return parsed
    mixed = _MIXED_RE.fullmatch(normalized)
    if mixed:
        return _parse_full_question(" ".join(mixed.group("question").split()))
    return None

def _visible_effective(
    statements: list[ParsedStatement],
    question: ParsedQuestion,
) -> ParsedStatement | None:
    matching = [
        item
        for item in statements
        if item.subject == question.subject
        and item.object == question.object
        and item.relation == question.relation
    ]
    return matching[-1] if matching else None


def decide(
    boundary: WindowBoundary,
    *,
    current_content: str,
) -> Decision:
    if not boundary.known:
        return Decision("boundary_unknown", "boundary_unknown")
    if not boundary.window_overflow:
        return Decision("direct", "window_not_overflow")
    if not boundary.hidden_teacher_exists:
        return Decision("direct", "no_hidden_teacher")

    text = " ".join(current_content.split())
    question = _parse_question(text)
    if question is None:
        if _SIMPLE_CONFIRM_RE.fullmatch(text):
            return Decision("direct", "simple_confirmation")
        if _TEACHING_RE.fullmatch(text):
            return Decision("direct", "ordinary_teaching")
        return Decision("clarify", "unsupported_pattern")

    if question.intent == "unsupported":
        return Decision("clarify", "unsupported_pattern", question=question)

    if not question.subject or not question.object:
        return Decision("clarify", "unsupported_pattern", question=question)

    statements = [
        parsed
        for message in boundary.visible_teacher
        if (parsed := parse_statement(message)) is not None
    ]
    effective = _visible_effective(statements, question)

    if question.intent == "history_compare":
        return Decision(
            "retrieve",
            "history_compare",
            query=f"{question.subject} {question.object}",
            question=question,
            visible_effective=effective,
        )

    # current/currently/still are not history markers by themselves.
    if effective is not None:
        return Decision(
            "direct",
            "visible_complete",
            question=question,
            visible_effective=effective,
        )
    return Decision(
        "retrieve",
        "hidden_retrieval",
        query=f"{question.subject} {question.object}",
        question=question,
    )
