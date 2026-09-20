"""HTTP contract tests for the history-review API."""

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import delete, func, select, update
from sqlalchemy import text as sa_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

import backend.main as main_module
from backend.chat_service import ChatService, SessionLockRegistry
from backend.database import (
    create_database_engine,
    create_tables,
    get_db,
    run_migrations,
)
from backend.exceptions import (
    HistoryReviewBoundaryUnavailable,
    HistoryReviewEventNotFound,
    HistoryReviewEventNotReviewable,
    HistoryReviewNoReviewableHistory,
    HistoryReviewRemoteAckRequired,
    HistoryReviewSessionNotFound,
    LLMError,
    SessionProfileConflictError,
    SessionProfileUnavailableError,
)
from backend.history_boundary import HISTORY_BOUNDARY_VERSION
from backend.history_review_stages import (
    HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT,
    HISTORY_REVIEW_STAGE_2_SYSTEM_PROMPT,
    STAGE1_RESERVED_OUTPUT_TOKENS,
    STAGE2_RESERVED_OUTPUT_TOKENS,
)
from backend.history_review_selection import HistoryReviewSourceMessageTooLarge
from backend.history_review_service import (
    HISTORY_REVIEW_EXECUTION_INTERRUPTED,
    HistoryReviewExecutionConflict,
    HistoryReviewExecutionNotFound,
    HistoryReviewService,
)
from backend.interaction_modes import CORRECTIVE_MODE, RECEIVE_TEACHING_MODE
from backend.llm_profiles import LLMProfile, LLMProfileRegistry
from backend.models import (
    ChatSession,
    HistoryReview,
    HistoryReviewFinding,
    HistoryReviewSource,
    Message,
    ModeSwitchEvent,
    utc_now,
)
from backend.schemas import ModeSwitchEventResponse

FAKE_MODEL = "fake"
LOCAL_MODEL = "local-model"
API_MODEL = "api-model"
FORBIDDEN_KEYS = {
    "called_llm",
    "raw_output",
    "raw_output_truncated",
    "attempt_count",
    "api_key",
    "base_url",
    "system_prompt",
    "reviewer_llm_profile_id_snapshot",
    "reviewer_llm_profile_kind_snapshot",
    "reviewer_llm_model_snapshot",
    "prompt_version_snapshot",
    "pipeline_version",
    "attempts",
    "budget_version_snapshot",
    "selection_policy_version",
    "remote_history_acknowledged",
    "remote_history_acknowledged_at",
    "remote_history_ack_message_count",
}


def _history_review_stage(messages) -> str | None:
    """Classify a request as chat, Stage 1, or Stage 2.

    Repair retries reuse the frozen Stage prompt but append a suffix
    after a newline, so a strict prefix boundary is sufficient.
    """
    if not messages:
        return None
    system = messages[0].get("content")
    if not isinstance(system, str):
        return None
    if (
        system == HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT
        or system.startswith(HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT + "\n")
    ):
        return "stage1"
    if (
        system == HISTORY_REVIEW_STAGE_2_SYSTEM_PROMPT
        or system.startswith(HISTORY_REVIEW_STAGE_2_SYSTEM_PROMPT + "\n")
    ):
        return "stage2"
    return None


_HISTORY_REVIEW_BASE_PROMPTS = {
    "stage1": HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT,
    "stage2": HISTORY_REVIEW_STAGE_2_SYSTEM_PROMPT,
}


def _default_stage1_response(messages) -> str:
    user_data = json.loads(messages[1]["content"])
    sources = []
    for source in user_data["sources"]:
        content = source["content"]
        quote = content.strip()[:256]
        claims = []
        if quote:
            claims.append({
                "claim_text": quote,
                "source_quote": quote,
            })
        sources.append({
            "source_message_id": source["source_message_id"],
            "claims": claims,
        })
    return json.dumps({
        "overflow": False,
        "sources": sources,
    }, ensure_ascii=False)


def _default_stage2_response(messages) -> str:
    user_data = json.loads(messages[1]["content"])
    return json.dumps({
        "claims": [
            {
                "claim_id": claim["claim_id"],
                "verdict": "correct",
                "correction_text": None,
                "explanation_text": None,
            }
            for claim in user_data["claims"]
        ],
    }, ensure_ascii=False)


class ApiSpyLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat_response = "chat reply"
        self.review_response: Any = None
        self.review_error: Exception | None = None
        self.chat_calls = 0
        self.review_calls = 0
        self.stage1_calls = 0
        self.stage2_calls = 0
        self.repair_calls = 0

    async def generate(
        self,
        messages,
        *,
        response_format=None,
        temperature=None,
        max_tokens=None,
    ):
        stage = _history_review_stage(messages)
        base_prompt = _HISTORY_REVIEW_BASE_PROMPTS.get(stage)
        is_repair = bool(
            base_prompt is not None
            and messages
            and messages[0]["content"] != base_prompt
        )
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "stage": stage,
            "repair": is_repair,
            "response_format": response_format,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        if stage is None:
            self.chat_calls += 1
            if callable(self.chat_response):
                return self.chat_response(messages)
            return self.chat_response

        self.review_calls += 1
        if stage == "stage1":
            self.stage1_calls += 1
        else:
            self.stage2_calls += 1
        if is_repair:
            self.repair_calls += 1

        if self.review_error is not None:
            raise self.review_error
        if self.review_response is not None:
            if callable(self.review_response):
                return self.review_response(messages)
            return self.review_response
        if stage == "stage1":
            return _default_stage1_response(messages)
        return _default_stage2_response(messages)


@dataclass
class ApiHarness:
    client: TestClient
    engine: Any
    session_factory: Any
    spy: ApiSpyLLM
    profiles: LLMProfileRegistry
    locks: SessionLockRegistry
    chat_service: ChatService
    history_review_service: HistoryReviewService


@pytest.fixture
def api_harness(tmp_path):
    engine = create_database_engine(f"sqlite:///{tmp_path / 'api.db'}")
    create_tables(bind=engine)
    run_migrations(engine)
    factory = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False
    )
    spy = ApiSpyLLM()
    profiles = LLMProfileRegistry([
        LLMProfile(
            id="default",
            label="Fake",
            kind="fake",
            model=FAKE_MODEL,
            client=spy,
            is_default=True,
        ),
        LLMProfile(
            id="local",
            label="Local",
            kind="local",
            model=LOCAL_MODEL,
            client=spy,
        ),
        LLMProfile(
            id="api",
            label="API",
            kind="api",
            model=API_MODEL,
            client=spy,
        ),
    ])
    locks = SessionLockRegistry()
    chat_service = ChatService(profiles=profiles, lock_registry=locks)
    review_service = HistoryReviewService(
        profiles=profiles,
        lock_registry=locks,
    )

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    main_module.app.dependency_overrides[get_db] = override_get_db
    original_chat = main_module.chat_service
    original_review = main_module.history_review_service
    main_module.chat_service = chat_service
    main_module.history_review_service = review_service
    try:
        with TestClient(main_module.app) as client:
            yield ApiHarness(
                client=client,
                engine=engine,
                session_factory=factory,
                spy=spy,
                profiles=profiles,
                locks=locks,
                chat_service=chat_service,
                history_review_service=review_service,
            )
    finally:
        main_module.app.dependency_overrides.pop(get_db, None)
        main_module.chat_service = original_chat
        main_module.history_review_service = original_review
        engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> int:
    response = client.post("/api/sessions", json={})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _send_message(
    client: TestClient,
    session_id: int,
    content: str = "teaching claim",
) -> dict:
    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"message": content},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _switch_mode(
    client: TestClient,
    session_id: int,
    mode: str,
) -> dict:
    response = client.patch(
        f"/api/sessions/{session_id}/interaction-mode",
        json={"interaction_mode": mode},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _switch_to_corrective(
    client: TestClient,
    session_id: int,
) -> dict:
    return _switch_mode(client, session_id, "corrective")


def _setup_reviewable_session(
    harness: ApiHarness,
    content: str = "teaching claim",
) -> tuple[int, dict, dict]:
    session_id = _create_session(harness.client)
    message_response = _send_message(harness.client, session_id, content)
    switch_body = _switch_to_corrective(harness.client, session_id)
    return session_id, message_response["user_message"], switch_body["switch_event"]


def _post_review(
    harness: ApiHarness,
    session_id: int,
    event_id: int,
    *,
    acknowledge: bool = False,
    retry_failed: bool = False,
):
    payload = {
        "mode_switch_event_id": event_id,
        "acknowledge_remote_history": acknowledge,
    }
    if retry_failed:
        payload["retry_failed"] = True
    return harness.client.post(
        f"/api/sessions/{session_id}/history-reviews",
        json=payload,
    )


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(value.keys())
        for item in value.values():
            keys |= _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= _all_keys(item)
    return keys


def _expected_summary(
    *, correct=0, incorrect=0, uncertain=0, no_claim=0,
):
    factual = correct + incorrect + uncertain
    return (
        f"Reviewed {factual} factual claims: {correct} correct, "
        f"{incorrect} incorrect, {uncertain} uncertain; {no_claim} "
        f"source(s) contained no verifiable factual claim."
    )


def _review_call_records(harness: ApiHarness) -> list[dict]:
    return [
        call for call in harness.spy.calls
        if call["stage"] is not None
    ]


def _seed_cross_profile_source(harness: ApiHarness) -> tuple[int, int]:
    """Seed an API reviewer review whose source message is local."""
    with harness.session_factory() as db:
        session = ChatSession(
            llm_profile_id="api",
            llm_model_snapshot=API_MODEL,
            interaction_mode="corrective",
        )
        db.add(session)
        db.flush()
        message = Message(
            session_id=session.id,
            role="user",
            content="local provenance claim",
            interaction_mode_snapshot="receive_teaching",
            llm_profile_id_snapshot="local",
            llm_profile_kind_snapshot="local",
            llm_model_snapshot=LOCAL_MODEL,
        )
        db.add(message)
        db.flush()
        event = ModeSwitchEvent(
            session_id=session.id,
            from_mode="receive_teaching",
            to_mode="corrective",
            history_boundary_version=HISTORY_BOUNDARY_VERSION,
            history_through_message_id=message.id,
            reviewable_user_message_count=1,
        )
        db.add(event)
        db.commit()
        return session.id, event.id


# ---------------------------------------------------------------------------
# POST
# ---------------------------------------------------------------------------


def test_post_creates_review_returns_201_and_detail(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] > 0
    assert body["session_id"] == session_id
    assert body["mode_switch_event_id"] == event["id"]
    assert body["status"] == "completed"
    assert body["findings_count"] == 1
    assert len(body["findings"]) == 1
    assert body["findings_count"] == len(body["findings"])
    assert body["findings"][0]["seq"] == 1
    assert body["findings"][0]["source_message_id"] == user_message["id"]
    assert body["findings"][0]["verdict"] == "correct"
    assert body["findings"][0]["claim_text"] == "teaching claim"
    assert body["summary"] == _expected_summary(correct=1)
    assert api_harness.spy.review_calls == 2
    assert api_harness.spy.stage1_calls == 1
    assert api_harness.spy.stage2_calls == 1


def test_post_reuses_review_returns_200_without_extra_llm(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)

    first = _post_review(api_harness, session_id, event["id"])
    second = _post_review(api_harness, session_id, event["id"])

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["findings"] == first.json()["findings"]
    assert api_harness.spy.review_calls == 2


def test_post_existing_pending_executes_without_creating_new_review(
    api_harness,
):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    first = _post_review(api_harness, session_id, event["id"])
    review_id = first.json()["id"]

    with api_harness.session_factory() as db:
        db.execute(
            update(HistoryReview)
            .where(HistoryReview.id == review_id)
            .values(
                status="pending",
                summary=None,
                coverage_note=None,
                raw_output=None,
                raw_output_truncated=False,
                error_code=None,
                error_message=None,
                started_at=None,
                completed_at=None,
            )
        )
        db.execute(
            delete(HistoryReviewFinding).where(
                HistoryReviewFinding.review_id == review_id
            )
        )
        db.commit()

    second = _post_review(api_harness, session_id, event["id"])

    assert second.status_code == 200, second.text
    assert second.json()["id"] == review_id, second.json()
    assert second.json()["status"] == "completed", second.json()
    assert second.json()["findings_count"] == 1
    assert api_harness.spy.review_calls == 4


def test_post_running_idempotent_without_llm(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()
    review_id = created["id"]

    with api_harness.session_factory() as db:
        db.execute(
            update(HistoryReview)
            .where(HistoryReview.id == review_id)
            .values(status="running", started_at=utc_now())
        )
        db.execute(
            delete(HistoryReviewFinding).where(
                HistoryReviewFinding.review_id == review_id
            )
        )
        db.commit()

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["findings"] == []
    assert api_harness.spy.review_calls == 2


def test_post_failed_idempotent_without_retry(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()
    review_id = created["id"]

    with api_harness.session_factory() as db:
        now = utc_now()
        db.execute(
            delete(HistoryReviewFinding).where(
                HistoryReviewFinding.review_id == review_id
            )
        )
        db.execute(
            update(HistoryReview)
            .where(HistoryReview.id == review_id)
            .values(
                status="failed",
                summary=None,
                coverage_note=None,
                raw_output=None,
                raw_output_truncated=False,
                error_code="history_review_llm_upstream_error",
                error_message="LLM request failed.",
                completed_at=now,
                updated_at=now,
            )
        )
        db.commit()

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_code"] == "history_review_llm_upstream_error"
    assert body["findings"] == []
    assert api_harness.spy.review_calls == 2


def test_post_execution_failed_is_normal_review_body(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    api_harness.spy.review_response = "not json"

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_code"] == (
        "history_review_extraction_json_syntax_error"
    )
    assert body["findings"] == []
    assert body["findings_count"] == 0
    assert api_harness.spy.review_calls == 2
    assert api_harness.spy.stage1_calls == 2
    assert api_harness.spy.stage2_calls == 0
    assert api_harness.spy.repair_calls == 1
    assert [
        (record["stage"], record["repair"])
        for record in _review_call_records(api_harness)
    ] == [("stage1", False), ("stage1", True)]


# ---------------------------------------------------------------------------
# GET
# ---------------------------------------------------------------------------


def test_get_list_returns_summaries_without_findings(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()

    response = api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews"
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["findings_count"] == 1
    assert "findings" not in body[0]
    assert body[0]["id"] == created["id"]
    assert api_harness.spy.review_calls == 2


def test_get_detail_returns_findings_and_matching_count(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()

    response = api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews/{created['id']}"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["findings_count"] == len(body["findings"])
    assert [f["seq"] for f in body["findings"]] == [1]
    assert body["findings"][0]["source_message_id"] == user_message["id"]
    assert api_harness.spy.review_calls == 2


def test_post_body_and_get_detail_match_after_refresh(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    posted = _post_review(api_harness, session_id, event["id"]).json()

    fetched = api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews/{posted['id']}"
    ).json()

    assert fetched == posted


def test_get_detail_other_session_is_404(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()
    other_session = _create_session(api_harness.client)

    response = api_harness.client.get(
        f"/api/sessions/{other_session}/history-reviews/{created['id']}"
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "history_review_not_found"


def test_get_does_not_call_llm(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()
    calls_before = api_harness.spy.review_calls

    api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews"
    )
    api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews/{created['id']}"
    )

    assert api_harness.spy.review_calls == calls_before


def test_list_sorted_by_created_at_then_id(api_harness):
    session_id = _create_session(api_harness.client)
    _send_message(api_harness.client, session_id, "first claim")
    first_event = _switch_to_corrective(
        api_harness.client, session_id
    )["switch_event"]
    first_review = _post_review(
        api_harness, session_id, first_event["id"]
    ).json()

    _switch_mode(api_harness.client, session_id, "receive_teaching")
    _send_message(api_harness.client, session_id, "second claim")
    second_event = _switch_to_corrective(
        api_harness.client, session_id
    )["switch_event"]
    second_review = _post_review(
        api_harness, session_id, second_event["id"]
    ).json()

    response = api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews"
    )
    ids = [row["id"] for row in response.json()]
    assert ids == [second_review["id"], first_review["id"]]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_session_id", [0, -1])
def test_positive_session_path_validation(api_harness, bad_session_id):
    response = api_harness.client.get(
        f"/api/sessions/{bad_session_id}/history-reviews"
    )
    assert response.status_code == 422


@pytest.mark.parametrize("bad_review_id", [0, -1])
def test_positive_review_path_validation(api_harness, bad_review_id):
    response = api_harness.client.get(
        f"/api/sessions/1/history-reviews/{bad_review_id}"
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"mode_switch_event_id": 0},
        {"mode_switch_event_id": True},
        {"mode_switch_event_id": "1"},
        {"mode_switch_event_id": 1, "acknowledge_remote_history": "false"},
        {"mode_switch_event_id": 1, "acknowledge_remote_history": 1},
        {"mode_switch_event_id": 1, "retry_failed": "true"},
        {"mode_switch_event_id": 1, "retry_failed": 1},
        {
            "mode_switch_event_id": 1,
            "acknowledge_remote_history": False,
            "extra": "forbidden",
        },
    ],
)
def test_create_request_validation(api_harness, payload):
    response = api_harness.client.post(
        "/api/sessions/1/history-reviews",
        json=payload,
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Exception mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exception", "status_code", "code"),
    [
        (HistoryReviewSessionNotFound(), 404, "history_review_session_not_found"),
        (HistoryReviewEventNotFound(), 404, "history_review_event_not_found"),
        (HistoryReviewEventNotReviewable(), 422, "history_review_event_not_reviewable"),
        (HistoryReviewBoundaryUnavailable(), 422, "history_review_boundary_unavailable"),
        (HistoryReviewNoReviewableHistory(), 422, "history_review_no_reviewable_history"),
        (
            HistoryReviewSourceMessageTooLarge(1, "too large"),
            422,
            "history_review_source_message_too_large",
        ),
        (
            SessionProfileUnavailableError("missing"),
            503,
            "history_review_reviewer_unavailable",
        ),
        (
            SessionProfileConflictError("model_changed", "snapshot"),
            409,
            "history_review_reviewer_conflict",
        ),
    ],
)
def test_prepare_exception_mapping(
    api_harness, monkeypatch, exception, status_code, code,
):
    async def _raise(*args, **kwargs):
        raise exception

    monkeypatch.setattr(
        api_harness.history_review_service,
        "prepare_history_review",
        _raise,
    )
    response = _post_review(api_harness, 1, 1)

    assert response.status_code == status_code
    assert response.json()["detail"]["code"] == code
    if code == "history_review_source_message_too_large":
        assert response.json()["detail"]["message"] == (
            "The reviewable message is too large to process."
        )


def test_ack_required_exception_envelope(api_harness, monkeypatch):
    async def _raise(*args, **kwargs):
        raise HistoryReviewRemoteAckRequired(
            source_message_count=2,
            reviewer_profile_label="API Model",
            reviewer_model="api-model",
            truncated=False,
        )

    monkeypatch.setattr(
        api_harness.history_review_service,
        "prepare_history_review",
        _raise,
    )
    response = _post_review(api_harness, 1, 1)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "history_review_remote_ack_required"
    assert detail["source_message_count"] == 2
    assert detail["reviewer_profile_label"] == "API Model"
    assert detail["reviewer_model"] == "api-model"
    assert detail["truncated"] is False
    assert not (set(detail) & FORBIDDEN_KEYS)


@pytest.mark.parametrize(
    ("exception", "expected_code"),
    [
        (HistoryReviewExecutionNotFound(), "history_review_not_found"),
        (
            HistoryReviewExecutionConflict(),
            "history_review_execution_conflict",
        ),
    ],
)
def test_execution_exception_mapping(
    api_harness, monkeypatch, exception, expected_code,
):
    async def _prepare(*args, **kwargs):
        return SimpleNamespace(
            created=True,
            review=SimpleNamespace(id=123),
        )

    async def _execute(*args, **kwargs):
        raise exception

    monkeypatch.setattr(
        api_harness.history_review_service,
        "prepare_history_review",
        _prepare,
    )
    monkeypatch.setattr(
        api_harness.history_review_service,
        "execute_history_review",
        _execute,
    )
    response = _post_review(api_harness, 1, 1)

    assert response.status_code in (404, 409)
    assert response.json()["detail"]["code"] == expected_code


def test_unexpected_database_error_returns_safe_500_and_rolls_back(
    api_harness, monkeypatch,
):
    async def _explode(*, session_id, mode_switch_event_id,
                       acknowledge_remote_history, db):
        db.add(ChatSession(
            title="rollback-marker",
            llm_profile_id="default",
            llm_model_snapshot=FAKE_MODEL,
        ))
        db.flush()
        raise SQLAlchemyError("secret SQL must not leak")

    monkeypatch.setattr(
        api_harness.history_review_service,
        "prepare_history_review",
        _explode,
    )
    response = _post_review(api_harness, 1, 1)

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == (
        "history_review_internal_error"
    )
    assert "secret SQL" not in response.text
    assert "Traceback" not in response.text

    with api_harness.session_factory() as db:
        count = db.execute(
            select(func.count())
            .select_from(ChatSession)
            .where(ChatSession.title == "rollback-marker")
        ).scalar()
    assert count == 0


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def test_ack_required_then_acknowledged(api_harness):
    session_id, event_id = _seed_cross_profile_source(api_harness)

    required = _post_review(api_harness, session_id, event_id)
    assert required.status_code == 409
    assert required.json()["detail"]["code"] == (
        "history_review_remote_ack_required"
    )

    accepted = _post_review(
        api_harness,
        session_id,
        event_id,
        acknowledge=True,
    )
    assert accepted.status_code == 201
    assert accepted.json()["status"] == "completed"
    assert accepted.json()["findings_count"] == 1
    assert api_harness.spy.review_calls == 2


# ---------------------------------------------------------------------------
# OpenAPI / dependency assembly
# ---------------------------------------------------------------------------


def test_openapi_records_create_reuse_and_error_envelopes(api_harness):
    schema = api_harness.client.get("/openapi.json").json()
    post = schema["paths"][
        "/api/sessions/{session_id}/history-reviews"
    ]["post"]
    responses = post["responses"]

    assert "200" in responses
    assert "201" in responses
    assert "404" in responses
    assert "409" in responses
    assert "422" in responses
    assert "500" in responses
    assert "503" in responses
    assert responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HistoryReviewDetailResponse"
    }
    assert responses["201"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HistoryReviewDetailResponse"
    }
    conflict_refs = responses["409"]["content"]["application/json"]["schema"]["anyOf"]
    assert {
        item["$ref"] for item in conflict_refs
    } == {
        "#/components/schemas/HistoryReviewErrorResponse",
        "#/components/schemas/HistoryReviewAckRequiredResponse",
    }
    validation_refs = responses["422"]["content"]["application/json"]["schema"]["anyOf"]
    assert {
        item["$ref"] for item in validation_refs
    } == {
        "#/components/schemas/HistoryReviewErrorResponse",
        "#/components/schemas/HTTPValidationError",
    }
    assert "/api/history-reviews/{review_id}" not in schema["paths"]


def test_production_services_share_dependencies():
    assert (
        main_module.chat_service._lock_registry
        is main_module.history_review_service._lock_registry
    )
    assert (
        main_module.chat_service._profiles
        is main_module.history_review_service._profiles
    )


# ---------------------------------------------------------------------------
# Sensitive-field leakage
# ---------------------------------------------------------------------------


def test_responses_do_not_expose_sensitive_fields(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()
    list_body = api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews"
    ).json()
    detail_body = api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews/{created['id']}"
    ).json()

    response_keys = _all_keys(created) | _all_keys(list_body) | _all_keys(detail_body)
    assert not (response_keys & FORBIDDEN_KEYS)


def test_error_response_does_not_expose_sensitive_fields(
    api_harness, monkeypatch,
):
    async def _raise(*args, **kwargs):
        raise HistoryReviewSessionNotFound()

    monkeypatch.setattr(
        api_harness.history_review_service,
        "prepare_history_review",
        _raise,
    )
    response = _post_review(api_harness, 1, 1)

    assert not (_all_keys(response.json()) & FORBIDDEN_KEYS)


# ---------------------------------------------------------------------------
# Mode-switch event discovery API
# ---------------------------------------------------------------------------


def _insert_event(
    harness: ApiHarness,
    session_id: int,
    *,
    from_mode=RECEIVE_TEACHING_MODE,
    to_mode=CORRECTIVE_MODE,
    created_at=None,
    history_through_message_id=1,
    history_boundary_version=HISTORY_BOUNDARY_VERSION,
    reviewable_user_message_count=1,
) -> int:
    with harness.session_factory() as db:
        event = ModeSwitchEvent(
            session_id=session_id,
            from_mode=from_mode,
            to_mode=to_mode,
            history_through_message_id=history_through_message_id,
            history_boundary_version=history_boundary_version,
            reviewable_user_message_count=reviewable_user_message_count,
        )
        if created_at is not None:
            event.created_at = created_at
        db.add(event)
        db.commit()
        return event.id


def _get_events(harness: ApiHarness, session_id: int) -> list[dict]:
    response = harness.client.get(
        f"/api/sessions/{session_id}/mode-switch-events"
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_mode_switch_events_missing_session_returns_structured_404(
    api_harness,
):
    response = api_harness.client.get(
        "/api/sessions/999999/mode-switch-events"
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == (
        "history_review_session_not_found"
    )


def test_mode_switch_events_empty_session_returns_empty_list(api_harness):
    session_id = _create_session(api_harness.client)
    assert _get_events(api_harness, session_id) == []


def test_mode_switch_events_order_created_at_desc_then_id_desc(
    api_harness,
):
    session_id = _create_session(api_harness.client)
    first = _insert_event(
        api_harness,
        session_id,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        history_through_message_id=1,
        reviewable_user_message_count=1,
    )
    second = _insert_event(
        api_harness,
        session_id,
        from_mode=CORRECTIVE_MODE,
        to_mode=RECEIVE_TEACHING_MODE,
        created_at=datetime(2026, 1, 3, 12, 0, 0),
        history_through_message_id=2,
        reviewable_user_message_count=None,
    )
    third = _insert_event(
        api_harness,
        session_id,
        created_at=datetime(2026, 1, 3, 12, 0, 0),
        history_through_message_id=3,
        reviewable_user_message_count=2,
    )

    rows = _get_events(api_harness, session_id)

    assert [row["id"] for row in rows] == [third, second, first]
    assert [row["review_supported"] for row in rows] == [True, False, True]


def test_mode_switch_events_review_supported_true(api_harness):
    session_id = _create_session(api_harness.client)
    event_id = _insert_event(api_harness, session_id)

    rows = _get_events(api_harness, session_id)

    assert len(rows) == 1
    assert rows[0]["id"] == event_id
    assert rows[0]["review_supported"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "from_mode": CORRECTIVE_MODE,
            "to_mode": RECEIVE_TEACHING_MODE,
            "reviewable_user_message_count": None,
        },
        {"history_boundary_version": None},
        {"history_boundary_version": "history-boundary-v0"},
        {"history_through_message_id": None},
        {"reviewable_user_message_count": None},
        {"reviewable_user_message_count": 0},
    ],
)
def test_mode_switch_events_review_supported_false_variants(
    api_harness, overrides,
):
    session_id = _create_session(api_harness.client)
    event_id = _insert_event(api_harness, session_id, **overrides)

    rows = _get_events(api_harness, session_id)

    event = next(row for row in rows if row["id"] == event_id)
    assert event["review_supported"] is False


def test_patch_switch_event_includes_review_supported(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    assert event["review_supported"] is True

    response = api_harness.client.patch(
        f"/api/sessions/{session_id}/interaction-mode",
        json={"interaction_mode": RECEIVE_TEACHING_MODE},
    )
    assert response.status_code == 200
    back_event = response.json()["switch_event"]
    assert back_event is not None
    assert back_event["review_supported"] is False


def test_same_mode_patch_has_no_event_and_does_not_add_row(api_harness):
    session_id = _create_session(api_harness.client)
    before = _get_events(api_harness, session_id)

    response = api_harness.client.patch(
        f"/api/sessions/{session_id}/interaction-mode",
        json={"interaction_mode": RECEIVE_TEACHING_MODE},
    )

    assert response.status_code == 200
    assert response.json()["switch_event"] is None
    assert _get_events(api_harness, session_id) == before == []


def test_mode_switch_events_do_not_leak_across_sessions(api_harness):
    session_a = _create_session(api_harness.client)
    session_b = _create_session(api_harness.client)
    event_a = _insert_event(api_harness, session_a)
    event_b = _insert_event(api_harness, session_b)

    rows_a = _get_events(api_harness, session_a)
    rows_b = _get_events(api_harness, session_b)

    assert [row["id"] for row in rows_a] == [event_a]
    assert [row["id"] for row in rows_b] == [event_b]
    assert all(row["session_id"] == session_a for row in rows_a)
    assert all(row["session_id"] == session_b for row in rows_b)


def test_mode_switch_events_do_not_expose_review_or_raw_fields(api_harness):
    session_id = _create_session(api_harness.client)
    _insert_event(api_harness, session_id)

    rows = _get_events(api_harness, session_id)
    keys = _all_keys(rows)

    assert not (keys & {"review_id", "review_status", "raw_output", "findings"})


def test_mode_switch_events_openapi_contract(api_harness):
    schema = api_harness.client.get("/openapi.json").json()
    operation = schema["paths"][
        "/api/sessions/{session_id}/mode-switch-events"
    ]["get"]
    responses = operation["responses"]

    assert {"200", "404", "422", "500"}.issubset(responses)
    assert responses["200"]["content"]["application/json"]["schema"][
        "items"
    ] == {"$ref": "#/components/schemas/ModeSwitchEventResponse"}
    assert responses["404"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HistoryReviewErrorResponse"
    }
    assert responses["500"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HistoryReviewErrorResponse"
    }

    component = schema["components"]["schemas"]["ModeSwitchEventResponse"]
    assert "review_supported" in component["required"]


def test_mode_switch_event_response_requires_explicit_boolean():
    with pytest.raises(ValidationError):
        ModeSwitchEventResponse(
            id=1,
            session_id=1,
            from_mode=RECEIVE_TEACHING_MODE,
            to_mode=CORRECTIVE_MODE,
            created_at=datetime(2026, 1, 1, 12, 0, 0),
            history_through_message_id=1,
            history_boundary_version=HISTORY_BOUNDARY_VERSION,
            reviewable_user_message_count=1,
        )

    with pytest.raises(ValidationError):
        ModeSwitchEventResponse(
            id=1,
            session_id=1,
            from_mode=RECEIVE_TEACHING_MODE,
            to_mode=CORRECTIVE_MODE,
            created_at=datetime(2026, 1, 1, 12, 0, 0),
            history_through_message_id=1,
            history_boundary_version=HISTORY_BOUNDARY_VERSION,
            reviewable_user_message_count=1,
            review_supported=1,
        )


def test_zero_claim_http_path_skips_stage2(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)

    def zero_claim_response(messages):
        assert _history_review_stage(messages) == "stage1"
        user_data = json.loads(messages[1]["content"])
        return json.dumps({
            "overflow": False,
            "sources": [
                {
                    "source_message_id": source["source_message_id"],
                    "claims": [],
                }
                for source in user_data["sources"]
            ],
        }, ensure_ascii=False)

    api_harness.spy.review_response = zero_claim_response

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["findings_count"] == 1
    assert body["findings"][0]["source_message_id"] == user_message["id"]
    assert body["findings"][0]["verdict"] == "not_a_claim"
    assert body["findings"][0]["claim_text"] == "teaching claim"
    assert api_harness.spy.review_calls == 1
    assert api_harness.spy.stage1_calls == 1
    assert api_harness.spy.stage2_calls == 0
    assert not (_all_keys(body) & FORBIDDEN_KEYS)


def test_chat_and_review_calls_are_classified_separately(api_harness):
    session_id = _create_session(api_harness.client)

    chat_body = _send_message(api_harness.client, session_id, "hello there")

    assert chat_body["assistant_message"]["content"] == (
        api_harness.spy.chat_response
    )
    assert api_harness.spy.chat_calls == 1
    assert api_harness.spy.review_calls == 0

    event = _switch_to_corrective(
        api_harness.client, session_id
    )["switch_event"]
    assert api_harness.spy.chat_calls == 1
    assert api_harness.spy.review_calls == 0

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 201, response.text
    assert response.json()["status"] == "completed"
    assert api_harness.spy.chat_calls == 1
    assert api_harness.spy.review_calls == 2
    assert api_harness.spy.stage1_calls == 1
    assert api_harness.spy.stage2_calls == 1
    assert [call["stage"] for call in api_harness.spy.calls] == [
        None,
        "stage1",
        "stage2",
    ]
    assert [call["repair"] for call in api_harness.spy.calls] == [
        False,
        False,
        False,
    ]


def test_api_review_call_options_match_pipeline(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 201, response.text
    review_records = _review_call_records(api_harness)
    assert [record["stage"] for record in review_records] == [
        "stage1",
        "stage2",
    ]
    assert all(record["repair"] is False for record in review_records)
    assert [record["response_format"] for record in review_records] == [
        {"type": "json_object"},
        {"type": "json_object"},
    ]
    assert [record["temperature"] for record in review_records] == [0, 0]
    assert review_records[0]["max_tokens"] == (
        STAGE1_RESERVED_OUTPUT_TOKENS
    )
    assert review_records[1]["max_tokens"] == (
        STAGE2_RESERVED_OUTPUT_TOKENS
    )
    assert api_harness.spy.review_calls == 2


def test_stage2_repair_success_end_to_end(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    stage2_attempts = []

    def respond(messages):
        stage = _history_review_stage(messages)
        assert stage is not None
        if stage == "stage1":
            return _default_stage1_response(messages)
        stage2_attempts.append(messages)
        if len(stage2_attempts) == 1:
            return "not json"
        user_data = json.loads(messages[1]["content"])
        return json.dumps({
            "claims": [
                {
                    "claim_id": claim["claim_id"],
                    "verdict": "correct",
                    "correction_text": None,
                    "explanation_text": None,
                }
                for claim in user_data["claims"]
            ],
        }, ensure_ascii=False)

    api_harness.spy.review_response = respond

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["findings_count"] == 1
    assert body["findings"][0]["verdict"] == "correct"
    assert api_harness.spy.review_calls == 3
    assert api_harness.spy.stage1_calls == 1
    assert api_harness.spy.stage2_calls == 2
    assert api_harness.spy.repair_calls == 1

    records = _review_call_records(api_harness)
    assert [(record["stage"], record["repair"]) for record in records] == [
        ("stage1", False),
        ("stage2", False),
        ("stage2", True),
    ]
    assert records[1]["messages"][1]["content"] == (
        records[2]["messages"][1]["content"]
    )
    assert records[2]["messages"][0]["content"].startswith(
        HISTORY_REVIEW_STAGE_2_SYSTEM_PROMPT + "\n"
    )
    assert not (_all_keys(body) & FORBIDDEN_KEYS)
    assert "raw_output" not in response.text


def test_review_llm_error_fails_on_first_call_without_retry(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    api_harness.spy.review_error = LLMError(
        detail="upstream boom", status_code=502,
    )

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_code"] == "history_review_llm_upstream_error"
    assert body["findings"] == []
    assert body["findings_count"] == 0
    assert api_harness.spy.review_calls == 1
    assert api_harness.spy.stage1_calls == 1
    assert api_harness.spy.stage2_calls == 0


def test_public_review_json_does_not_leak_audit_values(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()
    list_body = api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews"
    ).json()
    detail_body = api_harness.client.get(
        f"/api/sessions/{session_id}/history-reviews/{created['id']}"
    ).json()

    public_text = json.dumps(
        [created, list_body, detail_body], ensure_ascii=False
    )
    with api_harness.session_factory() as db:
        stored_raw = db.get(HistoryReview, created["id"]).raw_output

    assert stored_raw is not None
    assert stored_raw not in public_text
    assert "history-review-pipeline-v1" not in public_text
    assert HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT not in public_text
    assert HISTORY_REVIEW_STAGE_2_SYSTEM_PROMPT not in public_text
    assert "raw_output" not in public_text
    assert "attempts" not in public_text
    assert "api_key" not in public_text
    assert "base_url" not in public_text


def _mark_review_interrupted(harness, review_id, *, attempt_count=1):
    now = utc_now()
    with harness.session_factory() as db:
        db.execute(
            update(HistoryReview)
            .where(HistoryReview.id == review_id)
            .values(
                status="failed",
                error_code=HISTORY_REVIEW_EXECUTION_INTERRUPTED,
                error_message="History review execution was interrupted.",
                summary=None,
                coverage_note=None,
                raw_output=None,
                raw_output_truncated=False,
                attempt_count=attempt_count,
                started_at=now,
                completed_at=now,
                updated_at=now,
            )
        )
        db.execute(
            delete(HistoryReviewFinding).where(
                HistoryReviewFinding.review_id == review_id
            )
        )
        db.commit()


def _review_row(harness, review_id):
    with harness.session_factory() as db:
        return db.get(HistoryReview, review_id)


def _source_count(harness, review_id):
    with harness.session_factory() as db:
        return db.execute(
            select(func.count())
            .select_from(HistoryReviewSource)
            .where(HistoryReviewSource.review_id == review_id)
        ).scalar()


def _finding_count(harness, review_id):
    with harness.session_factory() as db:
        return db.execute(
            select(func.count())
            .select_from(HistoryReviewFinding)
            .where(HistoryReviewFinding.review_id == review_id)
        ).scalar()


def test_post_retry_no_existing_review_returns_409(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)

    response = _post_review(
        api_harness, session_id, event["id"], retry_failed=True,
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == (
        "history_review_retry_not_allowed"
    )
    assert api_harness.spy.review_calls == 0
    with api_harness.session_factory() as db:
        assert db.execute(
            select(func.count()).select_from(HistoryReview)
        ).scalar() == 0


def test_post_retry_other_failed_returns_409_and_keeps_state(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    api_harness.spy.review_response = "not json"
    first = _post_review(api_harness, session_id, event["id"])
    assert first.status_code == 201
    review_id = first.json()["id"]
    calls_before = api_harness.spy.review_calls
    row_before = _review_row(api_harness, review_id)

    response = _post_review(
        api_harness, session_id, event["id"], retry_failed=True,
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == (
        "history_review_retry_not_allowed"
    )
    assert api_harness.spy.review_calls == calls_before
    row_after = _review_row(api_harness, review_id)
    assert row_after.status == "failed"
    assert row_after.error_code == "history_review_extraction_json_syntax_error"
    assert row_after.attempt_count == row_before.attempt_count


def test_post_retry_interrupted_reuses_review_and_source_rows(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    first = _post_review(api_harness, session_id, event["id"])
    assert first.status_code == 201
    review_id = first.json()["id"]
    source_count_before = _source_count(api_harness, review_id)
    _mark_review_interrupted(api_harness, review_id)
    calls_before = api_harness.spy.review_calls

    response = _post_review(
        api_harness, session_id, event["id"], retry_failed=True,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == review_id
    assert body["status"] == "completed"
    assert body["findings_count"] == 1
    assert api_harness.spy.review_calls == calls_before + 2
    row = _review_row(api_harness, review_id)
    assert row.status == "completed"
    assert row.attempt_count == 2
    assert _source_count(api_harness, review_id) == source_count_before
    assert _finding_count(api_harness, review_id) == 1
    assert not (_all_keys(body) & FORBIDDEN_KEYS)
    with api_harness.session_factory() as db:
        stored_raw = db.get(HistoryReview, review_id).raw_output
    assert stored_raw is not None
    assert stored_raw not in response.text
    assert "history-review-pipeline-v1" not in response.text
    assert HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT not in response.text
    assert HISTORY_REVIEW_STAGE_2_SYSTEM_PROMPT not in response.text
    assert "raw_output" not in response.text
    assert "api_key" not in response.text
    assert "base_url" not in response.text


def test_post_retry_completed_is_noop(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    first = _post_review(api_harness, session_id, event["id"])
    assert first.status_code == 201
    review_id = first.json()["id"]
    calls_before = api_harness.spy.review_calls

    response = _post_review(
        api_harness, session_id, event["id"], retry_failed=True,
    )

    assert response.status_code == 200, response.text
    assert response.json()["id"] == review_id
    assert response.json()["status"] == "completed"
    assert api_harness.spy.review_calls == calls_before
    assert _review_row(api_harness, review_id).attempt_count == 1


def test_post_retry_running_and_pending_are_authoritative_noops(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    first = _post_review(api_harness, session_id, event["id"])
    review_id = first.json()["id"]

    with api_harness.session_factory() as db:
        db.execute(
            update(HistoryReview)
            .where(HistoryReview.id == review_id)
            .values(status="running", started_at=utc_now())
        )
        db.commit()
    calls_before = api_harness.spy.review_calls

    running = _post_review(
        api_harness, session_id, event["id"], retry_failed=True,
    )
    assert running.status_code == 200, running.text
    assert running.json()["status"] == "running"
    assert api_harness.spy.review_calls == calls_before

    with api_harness.session_factory() as db:
        db.execute(
            update(HistoryReview)
            .where(HistoryReview.id == review_id)
            .values(
                status="pending",
                started_at=None,
                completed_at=None,
                summary=None,
                coverage_note=None,
                raw_output=None,
                raw_output_truncated=False,
                error_code=None,
                error_message=None,
            )
        )
        db.commit()
    calls_before = api_harness.spy.review_calls

    pending = _post_review(
        api_harness, session_id, event["id"], retry_failed=True,
    )
    assert pending.status_code == 200, pending.text
    assert pending.json()["status"] == "pending"
    assert api_harness.spy.review_calls == calls_before


def test_get_list_and_detail_are_read_only(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()
    review_id = created["id"]
    calls_before = api_harness.spy.review_calls

    with api_harness.engine.begin() as conn:
        conn.execute(sa_text(
            "CREATE TRIGGER fail_review_update "
            "BEFORE UPDATE ON history_reviews "
            "BEGIN SELECT RAISE(FAIL, 'review update must not happen'); END"
        ))
    try:
        list_response = api_harness.client.get(
            f"/api/sessions/{session_id}/history-reviews"
        )
        detail_response = api_harness.client.get(
            f"/api/sessions/{session_id}/history-reviews/{review_id}"
        )
    finally:
        with api_harness.engine.begin() as conn:
            conn.execute(sa_text("DROP TRIGGER IF EXISTS fail_review_update"))

    assert list_response.status_code == 200, list_response.text
    assert detail_response.status_code == 200, detail_response.text
    assert api_harness.spy.review_calls == calls_before
    assert detail_response.json()["id"] == review_id


@pytest.mark.anyio
async def test_lifespan_recovers_abandoned_running_without_llm(
    api_harness, monkeypatch,
):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    created = _post_review(api_harness, session_id, event["id"]).json()
    review_id = created["id"]
    with api_harness.session_factory() as db:
        db.execute(
            update(HistoryReview)
            .where(HistoryReview.id == review_id)
            .values(
                status="running",
                error_code=None,
                error_message=None,
                started_at=utc_now(),
                completed_at=None,
            )
        )
        db.commit()
    calls_before = api_harness.spy.review_calls

    monkeypatch.setattr(main_module, "create_tables", lambda: None)
    monkeypatch.setattr(
        main_module, "run_migrations", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        main_module, "SessionLocal", api_harness.session_factory,
    )

    async with main_module.lifespan(main_module.app):
        pass

    row = _review_row(api_harness, review_id)
    assert row.status == "failed"
    assert row.error_code == HISTORY_REVIEW_EXECUTION_INTERRUPTED
    assert row.summary is None
    assert row.raw_output is None
    assert _finding_count(api_harness, review_id) == 0
    assert api_harness.spy.review_calls == calls_before


@pytest.mark.anyio
async def test_lifespan_recovery_failure_fails_fast(
    api_harness, monkeypatch,
):
    def _raise(db):
        raise SQLAlchemyError("simulated startup recovery failure")

    monkeypatch.setattr(
        api_harness.history_review_service,
        "recover_abandoned_running_reviews",
        _raise,
    )
    monkeypatch.setattr(main_module, "create_tables", lambda: None)
    monkeypatch.setattr(
        main_module, "run_migrations", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        main_module, "SessionLocal", api_harness.session_factory,
    )

    with pytest.raises(SQLAlchemyError, match="startup recovery failure"):
        async with main_module.lifespan(main_module.app):
            pass
