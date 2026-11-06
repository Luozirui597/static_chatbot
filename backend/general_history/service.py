"""Production orchestration for general teaching-history retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from backend.general_history.client import ProductionGeneralClient
from backend.general_history.engine import GeneralSearchEngine
from backend.interaction_modes import (
    RECEIVE_TEACHING_MODE,
    get_programmatic_system_prompt,
    get_system_prompt_for_mode,
)
from backend.programmatic_history.service import ProgrammaticHistoryService
from backend.teaching_strategies import get_teaching_prompt


@dataclass(frozen=True)
class GeneralHistorySettings:
    enabled: bool = False
    profile_id: str = ""
    model: str = ""
    decision_timeout: float = 3.0
    overall_timeout: float = 12.0


class GeneralHistoryService:
    def __init__(self, settings: GeneralHistorySettings) -> None:
        self.settings = settings

    @staticmethod
    def database_path(db: Session) -> Path | None:
        return ProgrammaticHistoryService.database_path(db)

    def applicability_reason(
        self,
        db: Session,
        *,
        profile,
        interaction_mode: str,
    ) -> str:
        if not self.settings.enabled:
            return "feature_disabled"
        if interaction_mode != RECEIVE_TEACHING_MODE:
            return "mode_not_receive"
        if getattr(profile, "kind", None) != "api":
            return "profile_not_api"
        if not self.settings.profile_id or not self.settings.model:
            return "general_config_missing"
        if getattr(profile, "id", None) != self.settings.profile_id:
            return "profile_mismatch"
        if getattr(profile, "model", None) != self.settings.model:
            return "model_mismatch"
        if self.database_path(db) is None:
            return "database_unsupported"
        return "enabled"

    def applicable(self, db: Session, *, profile, interaction_mode: str) -> bool:
        return (
            self.applicability_reason(
                db, profile=profile, interaction_mode=interaction_mode
            )
            == "enabled"
        )

    async def run(
        self,
        db: Session,
        *,
        session_id: int,
        current_message_id: int,
        profile,
        before_final=None,
        reply_language: str | None = None,
        teaching_strategy: str = "baseline",
    ):
        client = ProductionGeneralClient(
            profile.client,
            model=profile.model,
        )
        engine = GeneralSearchEngine(
            db,
            session_id=session_id,
            current_message_id=current_message_id,
            decision_client=client,
            final_client=client,
            database_path=self.database_path(db),
            decision_timeout=self.settings.decision_timeout,
            overall_timeout=self.settings.overall_timeout,
            final_system_prompt=get_teaching_prompt(teaching_strategy, history=True),
            clarify_system_prompt=get_teaching_prompt(teaching_strategy),
            defer_direct=True,
            before_final=before_final,
            reply_language=reply_language,
        )
        return await engine.run()
