"""Tests for the Iteration 2D-2 history-review execution service."""

import json
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy import text as sa_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

import backend.history_review_service as execution_module
from backend.database import create_database_engine, create_tables, run_migrations
from backend.exceptions import LLMError
from backend.history_review_prompt import HISTORY_REVIEW_PROMPT_VERSION
from backend.history_review_service import (
    FINALIZE_FAILED,
    INTERNAL_ERROR,
    LLM_EMPTY_RESPONSE,
    LLM_NON_STRING_RESPONSE,
    LLM_UPSTREAM_ERROR,
    PRIVACY_ACK_MISSING,
    PRIVACY_CONFLICT,
    PROMPT_VERSION_UNSUPPORTED,
    REVIEWER_KIND_CHANGED,
    REVIEWER_MODEL_CHANGED,
    REVIEWER_UNAVAILABLE,
    SOURCE_CORRUPTED,
    HistoryReviewExecutionConflict,
    HistoryReviewExecutionFinding,
    HistoryReviewExecutionNotFound,
    HistoryReviewExecutionResult,
    HistoryReviewService,
)
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

FAKE_MODEL = "fake"
LOCAL_MODEL = "local-model"
API_MODEL = "api-model"


class SpyClient:
    def __init__(self) -> None:
        self.messages: list[list[dict]] = []
        self.response = None
        self.error: Exception | None = None
        self.before_generate = None

    async def generate(self, messages):
        self.messages.append(messages)
        if self.before_generate is not None:
            self.before_generate()
        if self.error is not None:
            raise self.error
        if callable(self.response):
            return self.response(messages)
        return self.response


@pytest.fixture
def engine(tmp_path):
    eng = create_database_engine(f"sqlite:///{tmp_path / 'execution.db'}")
    create_tables(bind=eng)
    run_migrations(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    return sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False
    )


@pytest.fixture
def db(session_factory):
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def other_client():
    return SpyClient()


@pytest.fixture
def api_client():
    return SpyClient()


@pytest.fixture
def profiles(other_client, api_client):
    return LLMProfileRegistry([
        LLMProfile(
            id="default", label="Fake", kind="fake", model=FAKE_MODEL,
            client=other_client, is_default=True,
        ),
        LLMProfile(
            id="local", label="Local", kind="local", model=LOCAL_MODEL,
            client=other_client,
        ),
        LLMProfile(
            id="api", label="API", kind="api", model=API_MODEL,
            client=api_client,
        ),
    ])


@pytest.fixture
def lock_registry():
    from backend.chat_service import SessionLockRegistry

    return SessionLockRegistry()


@pytest.fixture
def service(profiles, lock_registry):
    return HistoryReviewService(
        profiles=profiles, lock_registry=lock_registry,
    )


def _make_session(db_session, profile_id="api", model_snapshot=API_MODEL):
    session = ChatSession(
        llm_profile_id=profile_id,
        llm_model_snapshot=model_snapshot,
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


def _make_message(
    db_session,
    session,
    content,
    *,
    profile_id="api",
    kind="api",
    model_snapshot=API_MODEL,
):
    message = Message(
        session_id=session.id,
        role="user",
        content=content,
        interaction_mode_snapshot="receive_teaching",
        llm_profile_id_snapshot=profile_id,
        llm_profile_kind_snapshot=kind,
        llm_model_snapshot=model_snapshot,
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(message)
    return message


def _make_event(db_session, session, messages):
    event = ModeSwitchEvent(
        session_id=session.id,
        from_mode="receive_teaching",
        to_mode="corrective",
        history_boundary_version="history-boundary-v1",
        history_through_message_id=messages[-1].id,
        reviewable_user_message_count=len(messages),
    )
    db_session.add(event)
    db_session.commit()
    db_session.refresh(event)
    return event


async def _prepare_pending(
    service,
    db,
    *,
    profile_id="api",
    kind="api",
    model_snapshot=API_MODEL,
    contents=("claim",),
    acknowledge_remote_history=False,
):
    session = _make_session(db, profile_id, model_snapshot)
    messages = tuple(
        _make_message(
            db,
            session,
            content,
            profile_id=profile_id,
            kind=kind,
            model_snapshot=model_snapshot,
        )
        for content in contents
    )
    event = _make_event(db, session, messages)
    result = await service.prepare_history_review(
        session_id=session.id,
        mode_switch_event_id=event.id,
        acknowledge_remote_history=acknowledge_remote_history,
        db=db,
    )
    return session, event, messages, result.review, result.sources


def _finding(
    source_message_id,
    verdict="correct",
    claim_text="claim",
    correction_text=None,
    explanation_text=None,
):
    return {
        "source_message_id": source_message_id,
        "verdict": verdict,
        "claim_text": claim_text,
        "correction_text": correction_text,
        "explanation_text": explanation_text,
    }


def _valid_response(messages, *, summary="summary", coverage_note=None):
    user_data = json.loads(messages[1]["content"])
    findings = []
    for source in user_data["sources"]:
        findings.append(_finding(source["source_message_id"]))
    return json.dumps(
        {
            "summary": summary,
            "coverage_note": coverage_note,
            "findings": findings,
        },
        ensure_ascii=False,
    )


def _refresh_review(db, review_id):
    db.expire_all()
    return db.execute(
        select(HistoryReview)
        .where(HistoryReview.id == review_id)
        .execution_options(populate_existing=True)
    ).scalars().first()


def _findings_for_review(db, review_id):
    db.expire_all()
    return list(db.execute(
        select(HistoryReviewFinding)
        .where(HistoryReviewFinding.review_id == review_id)
        .order_by(HistoryReviewFinding.seq.asc())
    ).scalars().all())


@pytest.mark.anyio
async def test_success_pending_to_completed_and_generate_once(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = lambda msgs: _valid_response(
        msgs, summary="finished", coverage_note="covered"
    )

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert isinstance(result, HistoryReviewExecutionResult)
    assert result.status == "completed"
    assert result.called_llm is True
    assert result.summary == "finished"
    assert result.coverage_note == "covered"
    assert result.error_code is None
    assert result.raw_output_truncated is False
    assert len(result.findings) == 1
    assert result.findings[0].seq == 1
    assert result.findings[0].source_message_id == messages[0].id
    assert result.findings[0].verdict == "correct"
    assert len(api_client.messages) == 1

    stored = _refresh_review(db, review.id)
    assert stored.status == "completed"
    assert stored.summary == "finished"
    assert stored.coverage_note == "covered"
    assert stored.error_code is None
    assert stored.started_at is not None
    assert stored.completed_at == stored.updated_at
    findings = _findings_for_review(db, review.id)
    assert [f.seq for f in findings] == [1]
    assert findings[0].source_message_id == messages[0].id
    assert findings[0].verdict == "correct"


@pytest.mark.anyio
async def test_generate_runs_outside_database_transaction(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    observed = []
    api_client.before_generate = lambda: observed.append(db.in_transaction())
    api_client.response = lambda msgs: _valid_response(msgs)

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "completed"
    assert observed == [False]


@pytest.mark.anyio
async def test_completed_idempotent_returns_findings_without_llm(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = lambda msgs: _valid_response(
        msgs, summary="saved"
    )
    first = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    assert first.status == "completed"
    assert len(api_client.messages) == 1

    second = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert second.status == "completed"
    assert second.called_llm is False
    assert second.summary == "saved"
    assert second.findings == first.findings
    assert len(api_client.messages) == 1


@pytest.mark.anyio
async def test_running_idempotent_returns_without_llm(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored = _refresh_review(db, review.id)
    stored.status = "running"
    stored.started_at = utc_now()
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "running"
    assert result.called_llm is False
    assert result.summary is None
    assert result.coverage_note is None
    assert result.findings == ()
    assert api_client.messages == []


@pytest.mark.anyio
async def test_failed_idempotent_returns_safe_error_without_llm(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored = _refresh_review(db, review.id)
    now = utc_now()
    stored.status = "failed"
    stored.error_code = REVIEWER_UNAVAILABLE
    stored.error_message = "Reviewer profile is unavailable."
    stored.completed_at = now
    stored.updated_at = now
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.called_llm is False
    assert result.summary is None
    assert result.coverage_note is None
    assert result.findings == ()
    assert result.error_code == REVIEWER_UNAVAILABLE
    assert result.error_message == "Reviewer profile is unavailable."
    assert api_client.messages == []


@pytest.mark.parametrize(
    ("session_id", "review_id"),
    [
        (True, 1),
        (0, 1),
        (-1, 1),
        ("1", 1),
        (1, True),
        (1, 0),
        (1, -1),
        (1, "1"),
    ],
)
@pytest.mark.anyio
async def test_invalid_inputs_are_rejected_before_llm(
    service, db, api_client, session_id, review_id,
):
    with pytest.raises(ValueError):
        await service.execute_history_review(
            session_id=session_id,
            review_id=review_id,
            db=db,
        )
    assert api_client.messages == []


@pytest.mark.anyio
async def test_missing_review_and_other_session_do_not_change_state(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    other_session = _make_session(db)

    with pytest.raises(HistoryReviewExecutionNotFound):
        await service.execute_history_review(
            session_id=session.id,
            review_id=99999,
            db=db,
        )
    with pytest.raises(HistoryReviewExecutionNotFound):
        await service.execute_history_review(
            session_id=other_session.id,
            review_id=review.id,
            db=db,
        )

    stored = _refresh_review(db, review.id)
    assert stored.status == "pending"
    assert api_client.messages == []


@pytest.mark.anyio
async def test_reviewer_unavailable_pending_to_failed(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored = _refresh_review(db, review.id)
    stored.reviewer_llm_profile_id_snapshot = "missing-reviewer"
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.called_llm is False
    assert result.error_code == REVIEWER_UNAVAILABLE
    assert result.findings == ()
    assert api_client.messages == []
    assert _assert_pending_failure_shape(db, review.id) is not None


def _assert_pending_failure_shape(db, review_id):
    stored = _refresh_review(db, review_id)
    assert stored.status == "failed"
    assert stored.started_at is None
    assert stored.completed_at is not None
    assert stored.completed_at == stored.updated_at
    assert stored.summary is None
    assert stored.coverage_note is None
    assert stored.raw_output is None
    assert stored.raw_output_truncated is False
    assert stored.error_code is not None
    assert stored.error_message is not None
    assert _findings_for_review(db, review_id) == []
    return stored


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("reviewer_llm_model_snapshot", "changed-model", REVIEWER_MODEL_CHANGED),
        ("reviewer_llm_profile_kind_snapshot", "local", REVIEWER_KIND_CHANGED),
        ("prompt_version_snapshot", "history-review-prompt-v0", PROMPT_VERSION_UNSUPPORTED),
    ],
)
@pytest.mark.anyio
async def test_pending_preflight_failures(
    service, db, api_client, field, value, expected_code,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored = _refresh_review(db, review.id)
    setattr(stored, field, value)
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.called_llm is False
    assert result.error_code == expected_code
    assert result.findings == ()
    assert api_client.messages == []
    _assert_pending_failure_shape(db, review.id)


@pytest.mark.anyio
async def test_source_missing_marks_preflight_failed(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    db.execute(
        delete(HistoryReviewSource).where(
            HistoryReviewSource.review_id == review.id
        )
    )
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == SOURCE_CORRUPTED
    assert result.called_llm is False
    assert api_client.messages == []


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("seq", SOURCE_CORRUPTED),
        ("role", SOURCE_CORRUPTED),
        ("mode", SOURCE_CORRUPTED),
        ("char_count", SOURCE_CORRUPTED),
        ("through", SOURCE_CORRUPTED),
    ],
)
@pytest.mark.anyio
async def test_source_corruption_variants(
    service, db, api_client, mutation, expected_code,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored_source = _refresh_sources(db, review.id)[0]
    if mutation == "seq":
        stored_source.seq = 2
    elif mutation == "role":
        message = db.get(Message, messages[0].id)
        message.role = "assistant"
    elif mutation == "mode":
        message = db.get(Message, messages[0].id)
        message.interaction_mode_snapshot = "corrective"
    elif mutation == "char_count":
        stored_source.content_char_count = 999999
    elif mutation == "through":
        stored = _refresh_review(db, review.id)
        stored.source_through_message_id = messages[0].id + 999
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == expected_code
    assert result.called_llm is False
    assert api_client.messages == []


@pytest.mark.anyio
async def test_extra_source_marks_preflight_failed(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    extra_message = _make_message(db, session, "extra source")
    db.add(HistoryReviewSource(
        review_id=review.id,
        session_id=session.id,
        message_id=extra_message.id,
        seq=2,
        content_char_count=len(extra_message.content),
    ))
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == SOURCE_CORRUPTED
    assert result.called_llm is False
    assert api_client.messages == []


@pytest.mark.anyio
async def test_out_of_order_sources_mark_preflight_failed(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db, contents=("one", "two"),
    )
    rows = _refresh_sources(db, review.id)
    db.execute(
        update(HistoryReviewSource)
        .where(HistoryReviewSource.id == rows[0].id)
        .values(seq=4)
    )
    db.execute(
        update(HistoryReviewSource)
        .where(HistoryReviewSource.id == rows[1].id)
        .values(seq=3)
    )
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == SOURCE_CORRUPTED
    assert result.called_llm is False
    assert api_client.messages == []


@pytest.mark.anyio
async def test_missing_message_and_lower_boundary_mismatch_are_corrupt(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    message = db.get(Message, messages[0].id)
    db.delete(message)
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    assert result.error_code == SOURCE_CORRUPTED

    db.expunge_all()

    # Restore a fresh pending review for the lower-boundary variant.
    session2, event2, messages2, review2, sources2 = (
        await _prepare_pending(service, db, contents=("restored",))
    )
    stored = _refresh_review(db, review2.id)
    stored.lower_bound_kind = "corrective_switch"
    stored.lower_bound_message_id = None
    db.commit()

    result2 = await service.execute_history_review(
        session_id=session2.id,
        review_id=review2.id,
        db=db,
    )
    assert result2.error_code == SOURCE_CORRUPTED
    assert api_client.messages == []


def _refresh_sources(db, review_id):
    db.expire_all()
    return list(db.execute(
        select(HistoryReviewSource)
        .where(HistoryReviewSource.review_id == review_id)
        .order_by(HistoryReviewSource.seq.asc())
        .execution_options(populate_existing=True)
    ).scalars().all())


@pytest.mark.anyio
async def test_privacy_ack_missing(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service,
        db,
        profile_id="api",
        kind="api",
        model_snapshot=API_MODEL,
        contents=("api claim",),
    )
    # Replace source provenance with a local message to force remote ack.
    message = db.get(Message, messages[0].id)
    message.llm_profile_id_snapshot = "local"
    message.llm_profile_kind_snapshot = "local"
    message.llm_model_snapshot = LOCAL_MODEL
    db.commit()
    stored = _refresh_review(db, review.id)
    stored.remote_history_acknowledged = False
    stored.remote_history_acknowledged_at = None
    stored.remote_history_ack_message_count = None
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == PRIVACY_ACK_MISSING
    assert result.called_llm is False
    assert api_client.messages == []


@pytest.mark.anyio
async def test_privacy_conflict(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    message = db.get(Message, messages[0].id)
    message.llm_profile_id_snapshot = "local"
    message.llm_profile_kind_snapshot = "local"
    message.llm_model_snapshot = LOCAL_MODEL
    db.commit()
    stored = _refresh_review(db, review.id)
    stored.remote_history_acknowledged = True
    stored.remote_history_acknowledged_at = utc_now()
    stored.remote_history_ack_message_count = 999
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == PRIVACY_CONFLICT
    assert result.called_llm is False
    assert api_client.messages == []


@pytest.mark.anyio
async def test_local_reviewer_privacy_contradiction(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service,
        db,
        profile_id="local",
        kind="local",
        model_snapshot=LOCAL_MODEL,
    )
    stored = _refresh_review(db, review.id)
    stored.remote_history_acknowledged = True
    stored.remote_history_acknowledged_at = utc_now()
    stored.remote_history_ack_message_count = 1
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == PRIVACY_CONFLICT
    assert result.called_llm is False
    assert api_client.messages == []


@pytest.mark.anyio
async def test_llm_error_is_persisted_as_failed(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.error = LLMError(detail="upstream boom", status_code=502)

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.called_llm is True
    assert result.error_code == LLM_UPSTREAM_ERROR
    assert result.findings == ()
    stored = _assert_running_failure_shape(db, review.id)
    assert stored.started_at is not None
    assert stored.raw_output is None
    assert stored.raw_output_truncated is False


def _assert_running_failure_shape(db, review_id):
    stored = _refresh_review(db, review_id)
    assert stored.status == "failed"
    assert stored.started_at is not None
    assert stored.completed_at is not None
    assert stored.completed_at == stored.updated_at
    assert stored.summary is None
    assert stored.coverage_note is None
    assert stored.raw_output_truncated in (False, True)
    assert _findings_for_review(db, review_id) == []
    return stored


@pytest.mark.anyio
async def test_non_string_output_is_failed(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = 123

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.called_llm is True
    assert result.error_code == LLM_NON_STRING_RESPONSE
    _assert_running_failure_shape(db, review.id)


@pytest.mark.anyio
async def test_blank_output_is_failed(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = "   \n\t"

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.called_llm is True
    assert result.error_code == LLM_EMPTY_RESPONSE
    _assert_running_failure_shape(db, review.id)


@pytest.mark.anyio
async def test_parser_syntax_error_is_preserved(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = "not json"

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == "history_review_json_syntax_error"
    stored = _assert_running_failure_shape(db, review.id)
    assert stored.raw_output == "not json"


@pytest.mark.anyio
async def test_parser_structure_and_semantic_codes_are_preserved(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )

    api_client.response = "[]"
    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    assert result.error_code == "history_review_json_structure_error"

    stored = _refresh_review(db, review.id)
    stored.status = "pending"
    stored.started_at = None
    stored.completed_at = None
    stored.error_code = None
    stored.error_message = None
    stored.raw_output = None
    stored.raw_output_truncated = False
    db.commit()

    api_client.response = json.dumps({
        "summary": "s",
        "coverage_note": None,
        "findings": [
            _finding(messages[0].id, "maybe")
        ],
    })
    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    assert result.error_code == "history_review_json_semantic_error"


@pytest.mark.anyio
async def test_parser_reference_and_uncovered_codes(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = json.dumps({
        "summary": "s",
        "coverage_note": None,
        "findings": [_finding(999999)],
    })
    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    assert result.error_code == "history_review_source_reference_invalid"


@pytest.mark.anyio
async def test_parser_uncovered_code_for_two_sources(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db, contents=("one", "two"),
    )
    api_client.response = json.dumps({
        "summary": "s",
        "coverage_note": None,
        "findings": [_finding(messages[0].id)],
    })

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == "history_review_source_uncovered"
    stored = _refresh_review(db, review.id)
    assert stored.raw_output is not None


@pytest.mark.anyio
async def test_raw_output_truncation_on_too_large(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    huge = "x" * 16001
    api_client.response = huge

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == "history_review_output_too_large"
    assert result.raw_output_truncated is True
    stored = _assert_running_failure_shape(db, review.id)
    assert stored.raw_output == "x" * 16000
    assert stored.raw_output_truncated is True


@pytest.mark.anyio
async def test_findings_keep_parser_original_order(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db, contents=("one", "two"),
    )
    api_client.response = json.dumps({
        "summary": "s",
        "coverage_note": None,
        "findings": [
            _finding(messages[0].id),
            _finding(messages[0].id, "uncertain", explanation_text="not sure"),
            _finding(messages[1].id, "incorrect", correction_text="fix", explanation_text="why"),
        ],
    })

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "completed"
    assert [f.seq for f in result.findings] == [1, 2, 3]
    assert [f.source_message_id for f in result.findings] == [
        messages[0].id,
        messages[0].id,
        messages[1].id,
    ]
    rows = _findings_for_review(db, review.id)
    assert [row.seq for row in rows] == [1, 2, 3]
    assert [row.source_message_id for row in rows] == [
        messages[0].id,
        messages[0].id,
        messages[1].id,
    ]


@pytest.mark.anyio
async def test_finalize_database_failure_then_failed_write_succeeds(
    service, db, engine, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = lambda msgs: _valid_response(
        msgs, summary="should not persist"
    )
    with engine.begin() as conn:
        conn.execute(sa_text(
            "CREATE TRIGGER fail_history_review_finding_insert "
            "BEFORE INSERT ON history_review_findings "
            "BEGIN SELECT RAISE(FAIL, 'simulated finding failure'); END"
        ))
    try:
        result = await service.execute_history_review(
            session_id=session.id,
            review_id=review.id,
            db=db,
        )
    finally:
        with engine.begin() as conn:
            conn.execute(sa_text(
                "DROP TRIGGER IF EXISTS fail_history_review_finding_insert"
            ))

    assert result.status == "failed"
    assert result.called_llm is True
    assert result.error_code == FINALIZE_FAILED
    stored = _assert_running_failure_shape(db, review.id)
    assert stored.summary is None
    assert stored.coverage_note is None
    assert stored.raw_output is not None
    assert _findings_for_review(db, review.id) == []


@pytest.mark.anyio
async def test_finalize_and_failed_write_both_fail_reraises(
    service, db, engine, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = lambda msgs: _valid_response(msgs)
    with engine.begin() as conn:
        conn.execute(sa_text(
            "CREATE TRIGGER fail_history_review_finding_insert "
            "BEFORE INSERT ON history_review_findings "
            "BEGIN SELECT RAISE(FAIL, 'simulated finding failure'); END"
        ))
        conn.execute(sa_text(
            "CREATE TRIGGER fail_history_review_failed_update "
            "BEFORE UPDATE ON history_reviews "
            "WHEN NEW.status = 'failed' "
            "BEGIN SELECT RAISE(FAIL, 'simulated failed write failure'); END"
        ))
    try:
        with pytest.raises(SQLAlchemyError):
            await service.execute_history_review(
                session_id=session.id,
                review_id=review.id,
                db=db,
            )
    finally:
        with engine.begin() as conn:
            conn.execute(sa_text(
                "DROP TRIGGER IF EXISTS fail_history_review_finding_insert"
            ))
            conn.execute(sa_text(
                "DROP TRIGGER IF EXISTS fail_history_review_failed_update"
            ))

    stored = _refresh_review(db, review.id)
    assert stored.status == "running"
    assert _findings_for_review(db, review.id) == []


@pytest.mark.anyio
async def test_source_selector_is_not_rerun(
    service, db, api_client, monkeypatch,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = lambda msgs: _valid_response(msgs)

    def _forbidden(*args, **kwargs):
        raise AssertionError("source selector must not run")

    monkeypatch.setattr(
        execution_module,
        "select_history_review_sources",
        _forbidden,
    )

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    assert result.status == "completed"


@pytest.mark.anyio
async def test_frozen_reviewer_does_not_fallback_to_current_session_model(
    service, db, api_client, other_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored_session = db.get(ChatSession, session.id)
    stored_session.llm_profile_id = "default"
    stored_session.llm_model_snapshot = FAKE_MODEL
    db.commit()
    api_client.response = lambda msgs: _valid_response(msgs)

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "completed"
    assert len(api_client.messages) == 1
    assert other_client.messages == []


@pytest.mark.anyio
async def test_result_dataclasses_are_frozen(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = lambda msgs: _valid_response(msgs)
    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    with pytest.raises(FrozenInstanceError):
        result.summary = "changed"
    with pytest.raises(FrozenInstanceError):
        result.findings[0].claim_text = "changed"


@pytest.mark.anyio
async def test_success_finalize_cas_miss_returns_authoritative_completed(
    service, db, api_client, session_factory,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )

    def switch_to_completed():
        other = session_factory()
        try:
            now = utc_now()
            other.execute(
                update(HistoryReview)
                .where(
                    HistoryReview.id == review.id,
                    HistoryReview.status == "running",
                )
                .values(
                    status="completed",
                    summary="authoritative summary",
                    coverage_note="authoritative coverage",
                    raw_output="authoritative raw",
                    raw_output_truncated=False,
                    error_code=None,
                    error_message=None,
                    completed_at=now,
                    updated_at=now,
                )
            )
            other.add(HistoryReviewFinding(
                review_id=review.id,
                seq=1,
                source_message_id=messages[0].id,
                verdict="correct",
                claim_text="authoritative claim",
                correction_text=None,
                explanation_text=None,
            ))
            other.commit()
        finally:
            other.close()

    api_client.before_generate = switch_to_completed
    api_client.response = lambda msgs: _valid_response(
        msgs, summary="this invocation"
    )

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "completed"
    assert result.called_llm is True
    assert result.summary == "authoritative summary"
    assert result.coverage_note == "authoritative coverage"
    assert [f.claim_text for f in result.findings] == [
        "authoritative claim"
    ]
    assert len(api_client.messages) == 1


@pytest.mark.anyio
async def test_running_failure_cas_miss_returns_authoritative_failed(
    service, db, api_client, session_factory,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )

    def switch_to_failed():
        other = session_factory()
        try:
            now = utc_now()
            other.execute(
                update(HistoryReview)
                .where(
                    HistoryReview.id == review.id,
                    HistoryReview.status == "running",
                )
                .values(
                    status="failed",
                    error_code=PRIVACY_CONFLICT,
                    error_message="Remote history acknowledgement is inconsistent.",
                    summary=None,
                    coverage_note=None,
                    raw_output=None,
                    raw_output_truncated=False,
                    completed_at=now,
                    updated_at=now,
                )
            )
            other.commit()
        finally:
            other.close()

    api_client.before_generate = switch_to_failed
    api_client.error = LLMError(detail="upstream boom", status_code=502)

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.called_llm is True
    assert result.error_code == PRIVACY_CONFLICT
    assert result.findings == ()
    assert len(api_client.messages) == 1


@pytest.mark.anyio
async def test_finalize_failed_cas_miss_authoritative_running_returns_running(
    service, db, engine, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = lambda msgs: _valid_response(msgs)

    with engine.begin() as conn:
        conn.execute(sa_text(
            "CREATE TRIGGER fail_history_review_finding_insert "
            "BEFORE INSERT ON history_review_findings "
            "BEGIN SELECT RAISE(FAIL, 'simulated finding failure'); END"
        ))
        conn.execute(sa_text(
            "CREATE TRIGGER ignore_history_review_failed_update "
            "BEFORE UPDATE ON history_reviews "
            "WHEN OLD.status = 'running' AND NEW.status = 'failed' "
            "BEGIN SELECT RAISE(IGNORE); END"
        ))
    try:
        result = await service.execute_history_review(
            session_id=session.id,
            review_id=review.id,
            db=db,
        )
    finally:
        with engine.begin() as conn:
            conn.execute(sa_text(
                "DROP TRIGGER IF EXISTS fail_history_review_finding_insert"
            ))
            conn.execute(sa_text(
                "DROP TRIGGER IF EXISTS ignore_history_review_failed_update"
            ))

    assert result.status == "running"
    assert result.called_llm is True
    assert result.summary is None
    assert result.coverage_note is None
    assert result.findings == ()
    assert len(api_client.messages) == 1

    stored = _refresh_review(db, review.id)
    assert stored.status == "running"
    assert _findings_for_review(db, review.id) == []


@pytest.mark.anyio
async def test_authoritative_reload_before_generate_has_called_llm_false(
    service, db, engine, api_client, monkeypatch,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )

    def _forbidden_build(*args, **kwargs):
        raise RuntimeError("simulated prompt build failure")

    monkeypatch.setattr(
        execution_module,
        "build_history_review_messages",
        _forbidden_build,
    )
    with engine.begin() as conn:
        conn.execute(sa_text(
            "CREATE TRIGGER ignore_history_review_failed_update "
            "BEFORE UPDATE ON history_reviews "
            "WHEN OLD.status = 'running' AND NEW.status = 'failed' "
            "BEGIN SELECT RAISE(IGNORE); END"
        ))
    try:
        result = await service.execute_history_review(
            session_id=session.id,
            review_id=review.id,
            db=db,
        )
    finally:
        with engine.begin() as conn:
            conn.execute(sa_text(
                "DROP TRIGGER IF EXISTS ignore_history_review_failed_update"
            ))

    assert result.status == "running"
    assert result.called_llm is False
    assert result.findings == ()
    assert api_client.messages == []
