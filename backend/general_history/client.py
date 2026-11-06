"""Production adapter for one decision and one final text generation."""

from __future__ import annotations

import asyncio
from typing import Any

from backend.exceptions import LLMInvalidResponseError
from backend.general_history.types import GeneralTransportEvent
from backend.llm_client import LLMClient
from backend.review_diagnostics import (
    LLMCallDiagnostics,
    clean_optional_int,
    error_category_value,
)


class ProductionGeneralClient:
    """Wrap the session's resolved production client without tools.

    ``http_sent`` means the selected client entered its local transport
    ``handle_async_request`` boundary.  It does not prove provider receipt.
    It is ``None`` when that client cannot expose transport evidence and is
    never inferred from "call attempted".
    """

    synthetic = False

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        decision_temperature: float = 0.0,
        decision_max_tokens: int = 800,
    ) -> None:
        self._client = client
        self.model = model
        self.decision_temperature = decision_temperature
        self.decision_max_tokens = decision_max_tokens
        self.events: list[GeneralTransportEvent] = []

    def _record(
        self,
        stage: str,
        diagnostics: LLMCallDiagnostics,
        status: str,
        *,
        pre_send: bool = False,
    ) -> None:
        sent = diagnostics.transport_started
        if sent is None and pre_send:
            sent = False
        reason = diagnostics.finish_reason
        if reason is not None and reason not in {
            "stop", "length", "tool_calls", "content_filter", "function_call"
        }:
            reason = "other"
        self.events.append(
            GeneralTransportEvent(
                stage=stage,
                status=status,
                http_sent=sent,
                upstream_http_status=clean_optional_int(
                    diagnostics.http_status, minimum=100, maximum=599
                ),
                error_category=error_category_value(diagnostics.error_category),
                finish_reason=reason,
            )
        )

    @staticmethod
    def _require_stop(
        stage: str,
        diagnostics: LLMCallDiagnostics,
    ) -> None:
        if diagnostics.finish_reason != "stop":
            raise LLMInvalidResponseError(
                detail="Upstream response did not complete normally",
                status_code=502,
                diagnostics=diagnostics,
            )

    async def generate_decision(
        self,
        messages: list[dict[str, Any]],
        *,
        deadline: float,
    ) -> str:
        diagnostics = LLMCallDiagnostics(stage="decision")
        try:
            raw = await self._client.generate(
                messages,
                response_format={"type": "json_object"},
                temperature=self.decision_temperature,
                max_tokens=self.decision_max_tokens,
                diagnostics=diagnostics,
            )
        except asyncio.CancelledError:
            self._record("decision", diagnostics, "cancelled")
            raise
        except ValueError:
            self._record("decision", diagnostics, "error", pre_send=True)
            raise
        except Exception:
            self._record("decision", diagnostics, "error")
            raise
        if diagnostics.finish_reason != "stop":
            self._record("decision", diagnostics, "invalid_finish")
            self._require_stop("decision", diagnostics)
        self._record("decision", diagnostics, "completed")
        return raw

    async def generate_final(
        self,
        messages: list[dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> str:
        diagnostics = LLMCallDiagnostics(stage="final")
        try:
            raw = await self._client.generate(
                messages,
                diagnostics=diagnostics,
            )
        except asyncio.CancelledError:
            self._record("final", diagnostics, "cancelled")
            raise
        except ValueError:
            self._record("final", diagnostics, "error", pre_send=True)
            raise
        except Exception:
            self._record("final", diagnostics, "error")
            raise
        if diagnostics.finish_reason != "stop":
            self._record("final", diagnostics, "invalid_finish")
            self._require_stop("final", diagnostics)
        self._record("final", diagnostics, "completed")
        return raw
