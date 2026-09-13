"""HTTP contract tests for the history-review API."""

import copy
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, update
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
    SessionProfileConflictError,
    SessionProfileUnavailableError,
)
from backend.history_boundary import HISTORY_BOUNDARY_VERSION
from backend.history_review_prompt import HISTORY_REVIEW_SYSTEM_PROMPT
from backend.history_review_selection import HistoryReviewSourceMessageTooLarge
from backend.history_review_service import (
    HistoryReviewExecutionConflict,
    HistoryReviewExecutionNotFound,
    HistoryReviewService,
)
from backend.llm_profiles import LLMProfile, LLMProfileRegistry
from backend.models import (
    ChatSession,
    HistoryReview,
    HistoryReviewFinding,
    Message,
    ModeSwitchEvent,
    utc_now,
)

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
    "budget_version_snapshot",
    "selection_policy_version",
    "remote_history_acknowledged",
    "remote_history_acknowledged_at",
    "remote_history_ack_message_count",
}


def _valid_review_json(messages) -> str:
    user_data = json.loads(messages[1]["content"])
    findings = []
    for source in user_data["sources"]:
        findings.append({
            "source_message_id": source["source_message_id"],
            "verdict": "correct",
            "claim_text": "claim text",
            "correction_text": None,
            "explanation_text": None,
        })
    return json.dumps({
        "summary": "review summary",
        "coverage_note": None,
        "findings": findings,
    }, ensure_ascii=False)


class ApiSpyLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.chat_response = "chat reply"
        self.review_response: Any = None
        self.review_error: Exception | None = None
        self.review_calls = 0

    async def generate(self, messages):
        self.calls.append(copy.deepcopy(messages))
        if messages and messages[0]["content"] == HISTORY_REVIEW_SYSTEM_PROMPT:
            self.review_calls += 1
            if self.review_error is not None:
                raise self.review_error
            if self.review_response is not None:
                if callable(self.review_response):
                    return self.review_response(messages)
                return self.review_response
            return _valid_review_json(messages)
        return self.chat_response


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
):
    return harness.client.post(
        f"/api/sessions/{session_id}/history-reviews",
        json={
            "mode_switch_event_id": event_id,
            "acknowledge_remote_history": acknowledge,
        },
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
    assert body["summary"] == "review summary"
    assert api_harness.spy.review_calls == 1


def test_post_reuses_review_returns_200_without_extra_llm(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)

    first = _post_review(api_harness, session_id, event["id"])
    second = _post_review(api_harness, session_id, event["id"])

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["findings"] == first.json()["findings"]
    assert api_harness.spy.review_calls == 1


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
    assert api_harness.spy.review_calls == 2


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
    assert api_harness.spy.review_calls == 1


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
    assert api_harness.spy.review_calls == 1


def test_post_execution_failed_is_normal_review_body(api_harness):
    session_id, user_message, event = _setup_reviewable_session(api_harness)
    api_harness.spy.review_response = "not json"

    response = _post_review(api_harness, session_id, event["id"])

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_code"] == "history_review_json_syntax_error"
    assert body["findings"] == []
    assert body["findings_count"] == 0


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
    assert api_harness.spy.review_calls == 1


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
