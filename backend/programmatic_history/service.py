"""Production-only programmatic retrieval for Receive teaching."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from backend.interaction_modes import (
    PROGRAMMATIC_HISTORY_PROMPT_VERSION,
    get_programmatic_system_prompt,
)
from backend.models import Message
from backend.programmatic_history.messages import build_programmatic_messages
from backend.programmatic_history.protocol import TOOL_NAME
from backend.programmatic_history.retrieval import ToolExecutor
from backend.programmatic_history.trigger import decide, load_window_boundary
from backend.programmatic_history.types import (
    ToolCall,
    ToolContext,
    ToolResult,
    ToolTimeout,
)

logger = logging.getLogger(__name__)


def _safe_log(method, message: str, *args) -> None:
    """Best-effort logging: observability must not change HTTP behavior."""
    try:
        method(message, *args)
    except Exception:
        pass


class ProgrammaticHistoryError(Exception):
    """A safe programmatic retrieval failure."""


class ProgrammaticHistoryTimeout(ProgrammaticHistoryError):
    """The read-only retrieval timed out."""


@dataclass(frozen=True)
class ProgrammaticSettings:
    enabled: bool = False
    profile_id: str = ""
    model: str = ""


@dataclass(frozen=True)
class ProgrammaticPlan:
    decision: Any
    boundary: Any
    retrieval: dict | None = None
    retrieval_status: str | None = None


class ProgrammaticHistoryService:
    def __init__(self, settings: ProgrammaticSettings) -> None:
        self.settings = settings

    @staticmethod
    def database_path(db: Session) -> Path | None:
        try:
            bind = db.get_bind()
            url = bind.url
            if url.get_backend_name() != "sqlite":
                return None
            if not url.database or url.database == ":memory:":
                return None
            path = Path(url.database).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            else:
                path = path.resolve()
            if not path.is_file():
                return None
            return path
        except Exception:
            return None

    def applicability_reason(
        self, db: Session, *, profile: Any, interaction_mode: str
    ) -> str:
        if not self.settings.enabled:
            return "feature_disabled"
        if interaction_mode != "receive_teaching":
            return "mode_not_receive"
        if profile.kind != "api":
            return "profile_not_api"
        if profile.id != self.settings.profile_id:
            return "profile_id_mismatch"
        if profile.model != self.settings.model:
            return "model_mismatch"
        if self.database_path(db) is None:
            return "database_unsupported"
        return "enabled"

    def applicable(self, db: Session, *, profile: Any, interaction_mode: str) -> bool:
        return self.applicability_reason(
            db, profile=profile, interaction_mode=interaction_mode
        ) == "enabled"

    def prepare(
        self,
        db: Session,
        *,
        session_id: int,
        current_message_id: int,
        content: str,
    ) -> ProgrammaticPlan:
        try:
            boundary = load_window_boundary(
                db,
                session_id=session_id,
                current_message_id=current_message_id,
            )
            decision = decide(boundary, current_content=content)
        except Exception as exc:
            _safe_log(logger.warning, 
                "programmatic_history decision_failed session_id=%s user_message_id=%s stage=decision error=%s",
                session_id,
                current_message_id,
                type(exc).__name__,
            )
            raise ProgrammaticHistoryError("decision_failed") from exc
        if not boundary.known:
            _safe_log(logger.warning, 
                "programmatic_history boundary_unknown session_id=%s user_message_id=%s",
                session_id,
                current_message_id,
            )
            return ProgrammaticPlan(decision=decision, boundary=boundary)
        return ProgrammaticPlan(decision=decision, boundary=boundary)

    async def retrieve_async(
        self,
        db: Session,
        *,
        session_id: int,
        current_message_id: int,
        plan: ProgrammaticPlan,
    ) -> ToolResult | None:
        if plan.decision.action != "retrieve":
            return None
        path = self.database_path(db)
        if path is None:
            raise ProgrammaticHistoryError("database_unsupported")
        call = ToolCall(
            id="programmatic_call_1",
            name=TOOL_NAME,
            arguments={"query": plan.decision.query or "", "limit": 3},
        )
        started = time.monotonic()
        try:
            result = await ToolExecutor(path).execute(
                call,
                ToolContext(
                    session_id=session_id,
                    current_message_id=current_message_id,
                ),
                before_id=plan.boundary.visible_start_id,
                deadline=time.monotonic() + 1.0,
            )
        except ToolTimeout as exc:
            _safe_log(logger.warning, 
                "programmatic_history retrieval_failed session_id=%s user_message_id=%s stage=retrieval_timeout error=%s latency_ms=%s",
                session_id,
                current_message_id,
                type(exc).__name__,
                int((time.monotonic() - started) * 1000),
            )
            raise ProgrammaticHistoryTimeout("retrieval_timeout") from exc
        except Exception as exc:
            _safe_log(logger.warning, 
                "programmatic_history retrieval_failed session_id=%s user_message_id=%s stage=retrieval_error error=%s latency_ms=%s",
                session_id,
                current_message_id,
                type(exc).__name__,
                int((time.monotonic() - started) * 1000),
            )
            raise ProgrammaticHistoryError("retrieval_error") from exc
        _safe_log(logger.info, 
            "programmatic_history retrieval_completed session_id=%s user_message_id=%s status=%s source_ids=%s latency_ms=%s",
            session_id,
            current_message_id,
            result.content.get("status"),
            [item.get("message_id") for item in result.content.get("results", []) if isinstance(item, dict)],
            int((time.monotonic() - started) * 1000),
        )
        return result

    def messages(
        self,
        db: Session,
        *,
        session_id: int,
        current_message_id: int,
        plan: ProgrammaticPlan,
        result: ToolResult,
    ) -> list[dict]:
        return build_programmatic_messages(
            db,
            session_id=session_id,
            current_message_id=current_message_id,
            system_prompt=get_programmatic_system_prompt(),
            boundary=plan.boundary,
            hidden_result=result.content,
            query=plan.decision.query or "",
        )
