"""Isolated FastAPI integration for the strategy-stability verifier.

The app owns its own SQLite database, sessions and messages.  It deliberately
does not import ``backend.main`` or the production message/router code.
"""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.reply_language import (
    ReplyLanguageInputError,
    ReplyLanguageMismatchError,
    validate_reply_language,
)
from experiments.strategy_stability.models import (
    ExperimentBase,
    ExperimentMessage,
    ExperimentSession,
)
from experiments.strategy_stability.verifier import (
    ContractError,
    GoalContext,
    InteractionState,
    Relation,
    TeacherMessage,
    TurnContext,
    ValidationResult,
    parse_candidate,
    parse_teacher_evidence,
    validate_candidate,
)

CONTRACT_VERSION = "strategy-stability-integration-v1"
VERIFIER_VERSION = "strategy-stability-verifier-v5"
BEHAVIOR_VERSION = "fake-candidate-v1"
SUPPORTED_STRATEGIES = frozenset({"standard", "a", "b", "ab"})
SUPPORTED_LANGUAGES = frozenset({"en", "zh"})
MAX_MESSAGES = 1000


class ExperimentInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CandidateRequest:
    """Frozen server-owned input handed to a candidate source.

    All nested values are immutable or tuples.  The fake records this object;
    a future adapter must use these frozen values instead of re-reading mutable
    session state.
    """

    session_id: int
    user_message_id: int
    question: str
    evidence: tuple[TeacherMessage, ...]
    turn_context: TurnContext
    strategy: str
    language: str
    model: str
    contract_version: str
    verifier_version: str
    behavior_version: str
    asked: int
    remediated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_message_id": self.user_message_id,
            "question": self.question,
            "evidence": [
                {"message_id": message.message_id, "content": message.content}
                for message in self.evidence
            ],
            "turn_context": _turn_to_dict(self.turn_context),
            "strategy": self.strategy,
            "language": self.language,
            "model": self.model,
            "contract_version": self.contract_version,
            "verifier_version": self.verifier_version,
            "behavior_version": self.behavior_version,
            "asked": self.asked,
            "remediated": self.remediated,
        }


class FakeCandidateModel:
    """Deterministic candidate source used by the isolated tests.

    The fake records only the number and identity of generation requests.  It
    never acts as a verifier: its output is always passed through the same
    ``validate_candidate`` path as any future model adapter.
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self.frozen_requests: list[CandidateRequest] = []
        self._lock = threading.Lock()
        self.before_response: Callable[[int], None] | None = None

    def generate(self, request: CandidateRequest) -> Any:
        with self._lock:
            self.calls += 1
            call_index = self.calls
            self.requests.append(request.to_dict())
            self.frozen_requests.append(request)
        if self.before_response is not None:
            self.before_response(call_index)
        with self._lock:
            item = self._responses.pop(0) if self._responses else None
        if isinstance(item, BaseException):
            raise item
        if item is None:
            raise RuntimeError("fake candidate response exhausted")
        return item


def _relation_dict(value: Relation) -> dict[str, str]:
    return {"subject": value.subject, "object": value.object}


def _relation_from_dict(value: dict[str, Any]) -> Relation:
    if not isinstance(value, dict):
        raise ExperimentInputError("invalid_relation", "relation must be an object")
    subject = value.get("subject")
    object_ = value.get("object")
    if not isinstance(subject, str) or not isinstance(object_, str):
        raise ExperimentInputError("invalid_relation", "relation fields must be strings")
    return Relation(subject, object_)


def _goal_to_dict(goal: GoalContext | None) -> dict[str, Any] | None:
    if goal is None:
        return None
    return {
        "original": _relation_dict(goal.original),
        "current_subgoals": [_relation_dict(r) for r in goal.current_subgoals],
        "askable_gaps": [_relation_dict(r) for r in goal.askable_gaps],
        "missing_relations": [_relation_dict(r) for r in goal.missing_relations],
        "missing_rule": goal.missing_rule,
        "gap_targets": [
            [_relation_dict(gap), _relation_dict(target)]
            for gap, target in goal.gap_targets
        ],
    }


def _goal_from_dict(raw: Any) -> GoalContext | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ExperimentInputError("invalid_goal", "goal must be an object or null")
    return GoalContext(
        original=_relation_from_dict(raw.get("original")),
        current_subgoals=tuple(
            _relation_from_dict(x) for x in raw.get("current_subgoals", [])
        ),
        askable_gaps=tuple(
            _relation_from_dict(x) for x in raw.get("askable_gaps", [])
        ),
        missing_relations=tuple(
            _relation_from_dict(x) for x in raw.get("missing_relations", [])
        ),
        missing_rule=bool(raw.get("missing_rule", False)),
        gap_targets=tuple(
            (_relation_from_dict(pair[0]), _relation_from_dict(pair[1]))
            for pair in raw.get("gap_targets", [])
        ),
    )


def _turn_to_dict(context: TurnContext) -> dict[str, Any]:
    return {
        "goal": _goal_to_dict(context.goal),
        "turn_kind": context.turn_kind,
        "allow_infer": context.allow_infer,
        "allow_ask": context.allow_ask,
        "allow_clarify": context.allow_clarify,
        "required_targets": [
            _relation_dict(r) for r in context.required_targets
        ],
    }


def _turn_from_dict(raw: Any) -> TurnContext:
    if not isinstance(raw, dict):
        raise ExperimentInputError("invalid_turn_context", "turn_context required")
    goal = _goal_from_dict(raw.get("goal"))
    if goal is None:
        raise ExperimentInputError("invalid_turn_context", "goal required")
    required = raw.get("required_targets", [])
    if not isinstance(required, list):
        raise ExperimentInputError("invalid_turn_context", "required_targets must be a list")
    return TurnContext(
        goal=goal,
        turn_kind=str(raw.get("turn_kind", "question")),
        allow_infer=bool(raw.get("allow_infer", True)),
        allow_ask=bool(raw.get("allow_ask", True)),
        allow_clarify=bool(raw.get("allow_clarify", False)),
        required_targets=tuple(_relation_from_dict(x) for x in required),
    )


def _evidence_from_raw(raw: Any, *, session_id: int) -> list[TeacherMessage]:
    if not isinstance(raw, list):
        raise ExperimentInputError("invalid_evidence", "evidence must be a list")
    messages: list[TeacherMessage] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ExperimentInputError("invalid_evidence", "evidence item must be an object")
        message_id = item.get("message_id")
        content = item.get("content")
        if not isinstance(message_id, int) or isinstance(message_id, bool):
            raise ExperimentInputError("invalid_evidence", "message_id must be an integer")
        if message_id in seen or message_id <= 0:
            raise ExperimentInputError("invalid_evidence", "message_id must be unique and positive")
        if not isinstance(content, str) or not content.strip():
            raise ExperimentInputError("invalid_evidence", "content required")
        seen.add(message_id)
        messages.append(TeacherMessage(message_id, session_id, "user", content))
    return messages


def _load_evidence(session: ExperimentSession) -> list[TeacherMessage]:
    raw = json.loads(session.evidence_json)
    return _evidence_from_raw(raw, session_id=session.id)


def _load_turn_context(session: ExperimentSession) -> TurnContext:
    return _turn_from_dict(json.loads(session.turn_context_json))


def _served_target_for_goal(goal: GoalContext | None, target: Relation) -> Relation:
    if goal is not None:
        for gap, served in goal.gap_targets:
            if gap == target:
                return served
        return goal.original
    return target


def _candidate_has_content_ask(raw: Any) -> bool:
    """True only for a content ask, not a server-authorized clarify."""
    normalized = parse_candidate(raw)
    return any(item.get("type") == "ask" for item in normalized["items"])


def _pending_from_candidate(raw: Any, context: TurnContext) -> dict[str, Any] | None:
    """Return a pending contract from the same normalized parser as validation."""
    normalized = parse_candidate(raw)
    asks = [
        item["ask"]
        for item in normalized["items"]
        if item.get("type") == "ask"
    ]
    if not asks:
        return None
    ask = asks[-1]
    target = Relation(ask["subject"], ask["object"])
    served = _served_target_for_goal(context.goal, target)
    return {
        "kind": ask["kind"],
        "target": {"subject": target.subject, "object": target.object},
        "served_target": {"subject": served.subject, "object": served.object},
    }


class ExperimentService:
    def __init__(
        self,
        database_path: str | Path,
        candidate_model: FakeCandidateModel,
    ) -> None:
        self.database_path = str(Path(database_path))
        self.engine = create_engine(
            f"sqlite:///{Path(self.database_path).resolve()}",
            connect_args={"check_same_thread": False},
        )
        ExperimentBase.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False
        )
        self.candidate_model = candidate_model
        self._locks: dict[int, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _session_lock(self, session_id: int) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[session_id] = lock
            return lock

    def create_session(
        self,
        *,
        strategy: str,
        language: str,
        evidence: Any,
        turn_context: Any,
        model_name: str,
    ) -> dict[str, Any]:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ExperimentInputError("invalid_strategy", "unsupported strategy")
        if language not in SUPPORTED_LANGUAGES:
            raise ExperimentInputError("invalid_language", "unsupported language")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ExperimentInputError("invalid_model", "model_name required")
        context = _turn_from_dict(turn_context)
        with self.SessionLocal.begin() as db:
            session = ExperimentSession(
                strategy=strategy,
                language=language,
                contract_version=CONTRACT_VERSION,
                verifier_version=VERIFIER_VERSION,
                behavior_version=BEHAVIOR_VERSION,
                model_name=model_name,
                evidence_json="[]",
                turn_context_json=json.dumps(_turn_to_dict(context), ensure_ascii=False),
            )
            db.add(session)
            db.flush()
            messages = _evidence_from_raw(evidence, session_id=session.id)
            session.evidence_json = json.dumps(
                [{"message_id": m.message_id, "content": m.content} for m in messages],
                ensure_ascii=False,
            )
            db.flush()
            return _session_payload(session)

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with self.SessionLocal() as db:
            session = db.get(ExperimentSession, session_id)
            if session is None:
                return None
            return _session_payload(session, db=db)

    def send(self, session_id: int, content: str) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ExperimentInputError("invalid_message", "message required")
        with self._session_lock(session_id):
            try:
                with self.SessionLocal.begin() as db:
                    session = db.get(ExperimentSession, session_id)
                    if session is None:
                        raise ExperimentInputError("session_not_found", "session not found")
                    if session.closed:
                        raise ExperimentInputError("interaction_closed", "interaction closed")
                    user = _message(
                        session,
                        role="user",
                        content=content,
                    )
                    db.add(user)
                    db.flush()
                    user_id = user.id
            except ExperimentInputError:
                raise
            except Exception:
                return _failure(
                    "user_store_failed",
                    "user_message_store_failed",
                    user_id=None,
                    user_saved=False,
                    status_code=500,
                )

            with self.SessionLocal() as db:
                session = db.get(ExperimentSession, session_id)
                evidence = parse_teacher_evidence(
                    _load_evidence(session),
                    session_id=session_id,
                    allowed_ids={m.message_id for m in _load_evidence(session)},
                )
                state = InteractionState(
                    asked=session.asked,
                    remediated=session.remediated,
                    closed=bool(session.closed),
                )
                context = _load_turn_context(session)
                frozen_evidence = tuple(
                    evidence.messages[mid] for mid in sorted(evidence.messages)
                )
                request = CandidateRequest(
                    session_id=session_id,
                    user_message_id=user_id,
                    question=content,
                    evidence=frozen_evidence,
                    turn_context=context,
                    strategy=session.strategy,
                    language=session.language,
                    model=session.model_name,
                    contract_version=session.contract_version,
                    verifier_version=session.verifier_version,
                    behavior_version=session.behavior_version,
                    asked=session.asked,
                    remediated=session.remediated,
                )
                try:
                    raw = self.candidate_model.generate(request)
                except Exception:
                    return _failure(
                        "candidate_generation_failed",
                        "candidate_generation_failed",
                        user_id=user_id,
                        status_code=502,
                    )
                result = validate_candidate(
                    raw,
                    strategy=session.strategy,
                    evidence=evidence,
                    state=state,
                    turn_context=context,
                )
                if not result.accepted:
                    return _failure(
                        "candidate_rejected",
                        result.code or "candidate_rejected",
                        user_id=user_id,
                        validation=result,
                        status_code=502,
                    )
                assert result.final_text is not None
                try:
                    pending = _pending_from_candidate(raw, context)
                except ContractError:
                    return _failure(
                        "candidate_state_invalid",
                        "candidate_state_invalid",
                        user_id=user_id,
                        validation=result,
                        status_code=502,
                    )
                has_content_ask = _candidate_has_content_ask(raw)
                if (
                    result.next_state.asked > state.asked
                    and has_content_ask
                    and pending is None
                ):
                    return _failure(
                        "candidate_state_invalid",
                        "candidate_state_invalid",
                        user_id=user_id,
                        validation=result,
                        status_code=502,
                    )
                try:
                    validate_reply_language(
                        result.final_text,
                        target=session.language,
                        current_text=content,
                        evidence_texts=tuple(
                            m.content for m in evidence.messages.values()
                        ),
                    )
                except (ReplyLanguageMismatchError, ReplyLanguageInputError) as exc:
                    return _failure(
                        "language_rejected",
                        "receive_reply_language_mismatch",
                        user_id=user_id,
                        validation=result,
                        status_code=502,
                        detail=str(exc),
                    )
                try:
                    with self.SessionLocal.begin() as db:
                        current = db.get(ExperimentSession, session_id)
                        if current is None:
                            raise RuntimeError("session disappeared")
                        db.add(
                            _message(
                                current,
                                role="assistant",
                                content=result.final_text,
                            )
                        )
                        current.asked = result.next_state.asked
                        current.remediated = result.next_state.remediated
                        current.closed = 1 if result.next_state.closed else 0
                        # A direct/infer answer does not count as remediation.
                        # Only a validated ask opens a pending gap, and only
                        # the explicit remediation endpoint clears it.
                        if pending is not None:
                            current.pending_gap_json = json.dumps(
                                pending, ensure_ascii=False
                            )
                        db.flush()
                except Exception:
                    return _failure(
                        "persistence_failed",
                        "assistant_state_commit_failed",
                        user_id=user_id,
                        validation=result,
                        status_code=500,
                    )
                return {
                    "status": "ok",
                    "user_message_id": user_id,
                    "assistant_text": result.final_text,
                    "validation": {
                        "status": result.status,
                        "source_ids": list(result.used_source_ids),
                    },
                    "asked": result.next_state.asked,
                    "remediated": result.next_state.remediated,
                    "pending_gap": pending,
                    "model_calls": self.candidate_model.calls,
                }

    def apply_remediation(
        self,
        session_id: int,
        *,
        gap: Any,
        teacher_message_id: int,
        content: str,
    ) -> dict[str, Any]:
        if not isinstance(teacher_message_id, int) or isinstance(teacher_message_id, bool):
            raise ExperimentInputError("invalid_evidence", "teacher_message_id required")
        if teacher_message_id <= 0:
            raise ExperimentInputError("invalid_evidence", "teacher_message_id required")
        if not isinstance(content, str) or not content.strip():
            raise ExperimentInputError("invalid_evidence", "content required")
        with self._session_lock(session_id):
            with self.SessionLocal.begin() as db:
                session = db.get(ExperimentSession, session_id)
                if session is None:
                    raise ExperimentInputError("session_not_found", "session not found")
                if session.pending_gap_json is None:
                    raise ExperimentInputError("no_pending_gap", "no pending gap")
                if session.remediated:
                    raise ExperimentInputError("already_remediated", "already remediated")
                pending = json.loads(session.pending_gap_json)
                target_raw = pending.get("target")
                served_raw = pending.get("served_target")
                kind = pending.get("kind")
                if not isinstance(target_raw, dict) or not isinstance(served_raw, dict):
                    raise ExperimentInputError("invalid_pending_gap", "pending gap is invalid")
                target = _relation_from_dict(target_raw)
                served = _relation_from_dict(served_raw)
                requested = _relation_from_dict(gap)
                if requested != target:
                    raise ExperimentInputError(
                        "remediation_mismatch",
                        "remediation does not match pending gap",
                    )

                evidence = json.loads(session.evidence_json)
                if any(int(item["message_id"]) == teacher_message_id for item in evidence):
                    raise ExperimentInputError(
                        "invalid_evidence", "duplicate teacher message id"
                    )

                parsed = parse_teacher_evidence(
                    [TeacherMessage(teacher_message_id, session_id, "user", content)],
                    session_id=session_id,
                    allowed_ids={teacher_message_id},
                )
                if parsed.unsupported_ids:
                    raise ExperimentInputError(
                        "remediation_unsupported",
                        "remediation content is not controlled teaching",
                    )

                if kind in ("missing_fact", "direct_target"):
                    fact = parsed.facts.get((target.subject, target.object))
                    if fact is None or fact.source_message_id != teacher_message_id:
                        raise ExperimentInputError(
                            "remediation_kind_mismatch",
                            "remediation must teach the requested relation",
                        )
                elif kind == "missing_rule":
                    rule = next(
                        (
                            value
                            for value in parsed.rules.values()
                            if value.rule_id == teacher_message_id
                        ),
                        None,
                    )
                    # The rule's conclusion must be the served target; a
                    # plain fact is not a rule and cannot satisfy this ask.
                    if rule is None or (rule.x, rule.z) != (
                        served.subject,
                        served.object,
                    ):
                        raise ExperimentInputError(
                            "remediation_kind_mismatch",
                            "remediation must provide the requested rule",
                        )
                else:
                    raise ExperimentInputError(
                        "invalid_pending_gap", "unsupported pending ask kind"
                    )

                evidence.append(
                    {"message_id": teacher_message_id, "content": content}
                )
                session.evidence_json = json.dumps(evidence, ensure_ascii=False)
                session.remediated = 1
                session.pending_gap_json = None
                db.flush()
                return {
                    "status": "ok",
                    "remediated": 1,
                    "evidence_message_id": teacher_message_id,
                }

    def close(self) -> None:
        self.engine.dispose()


def _message(session: ExperimentSession, *, role: str, content: str) -> ExperimentMessage:
    return ExperimentMessage(
        session_id=session.id,
        role=role,
        content=content,
        strategy_snapshot=session.strategy,
        language_snapshot=session.language,
        model_snapshot=session.model_name,
        contract_version_snapshot=session.contract_version,
        verifier_version_snapshot=session.verifier_version,
        behavior_version_snapshot=session.behavior_version,
    )


def _session_payload(
    session: ExperimentSession, *, db: Session | None = None
) -> dict[str, Any]:
    messages = []
    if db is not None:
        messages = [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "strategy": m.strategy_snapshot,
                "language": m.language_snapshot,
                "model": m.model_snapshot,
                "contract_version": m.contract_version_snapshot,
                "verifier_version": m.verifier_version_snapshot,
                "behavior_version": m.behavior_version_snapshot,
            }
            for m in db.scalars(
                select(ExperimentMessage)
                .where(ExperimentMessage.session_id == session.id)
                .order_by(ExperimentMessage.id)
            )
        ]
    return {
        "id": session.id,
        "strategy": session.strategy,
        "language": session.language,
        "model": session.model_name,
        "contract_version": session.contract_version,
        "verifier_version": session.verifier_version,
        "behavior_version": session.behavior_version,
        "asked": session.asked,
        "remediated": session.remediated,
        "closed": bool(session.closed),
        "pending_gap": (
            json.loads(session.pending_gap_json)
            if session.pending_gap_json is not None
            else None
        ),
        "messages": messages,
    }


def _failure(
    stage: str,
    code: str,
    *,
    user_id: int | None,
    validation: ValidationResult | None = None,
    status_code: int = 502,
    detail: str = "",
    user_saved: bool = True,
) -> JSONResponse:
    message = (
        "The user message was saved; no assistant message was saved."
        if user_saved
        else "The user message was not saved; no model call was made."
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "stage": stage,
            "code": code,
            "message": message,
            "user_message_id": user_id,
            "validation_status": validation.status if validation else None,
            "validation_code": validation.code if validation else None,
            "detail": detail,
        },
    )


def create_app(
    database_path: str | Path,
    candidate_model: FakeCandidateModel | None = None,
) -> FastAPI:
    model = candidate_model or FakeCandidateModel()
    service = ExperimentService(database_path, model)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="Strategy Stability Integration", lifespan=lifespan)
    app.state.service = service
    app.state.candidate_model = model

    @app.post("/sessions")
    def create_session(body: dict[str, Any]) -> Any:
        try:
            return service.create_session(
                strategy=body.get("strategy", ""),
                language=body.get("language", "en"),
                evidence=body.get("evidence", []),
                turn_context=body.get("turn_context"),
                model_name=body.get("model_name", "fake-candidate-v1"),
            )
        except ExperimentInputError as exc:
            return JSONResponse(
                status_code=422,
                content={"status": "error", "code": exc.code, "message": exc.message},
            )

    @app.get("/sessions/{session_id}")
    def get_session(session_id: int) -> Any:
        payload = service.get_session(session_id)
        if payload is None:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "code": "session_not_found"},
            )
        return payload

    @app.post("/sessions/{session_id}/messages")
    def send_message(session_id: int, body: dict[str, Any]) -> Any:
        try:
            return service.send(session_id, body.get("content", ""))
        except ExperimentInputError as exc:
            return JSONResponse(
                status_code=422,
                content={"status": "error", "code": exc.code, "message": exc.message},
            )
        except Exception:
            return JSONResponse(
                status_code=502,
                content={
                    "status": "error",
                    "code": "candidate_generation_failed",
                    "message": "The user message was saved; candidate generation failed.",
                },
            )

    @app.post("/sessions/{session_id}/remediation")
    def apply_remediation(session_id: int, body: dict[str, Any]) -> Any:
        try:
            return service.apply_remediation(
                session_id,
                gap=body.get("gap"),
                teacher_message_id=body.get("teacher_message_id", 0),
                content=body.get("content", ""),
            )
        except ExperimentInputError as exc:
            return JSONResponse(
                status_code=422,
                content={"status": "error", "code": exc.code, "message": exc.message},
            )

    return app
