"""Tests for the Iteration 2D-2 history-review execution service."""

import asyncio
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
from backend.history_review_pipeline import HistoryReviewPipelineError
from backend.history_review_stages import (
    HISTORY_REVIEW_PIPELINE_VERSION,
    HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT,
    HISTORY_REVIEW_STAGE_2_SYSTEM_PROMPT,
    STAGE1_RESERVED_OUTPUT_TOKENS,
    STAGE2_RESERVED_OUTPUT_TOKENS,
)
from backend.history_review_service import (
    FINALIZE_FAILED,
    HISTORY_REVIEW_EXECUTION_INTERRUPTED,
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
    HistoryReviewRetryNotAllowed,
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
        self.calls: list[dict] = []
        self.response = None
        self.error: Exception | None = None
        self.before_generate = None

    async def generate(
        self,
        messages,
        *,
        response_format=None,
        temperature=None,
        max_tokens=None,
    ):
        self.messages.append(messages)
        self.calls.append({
            "messages": messages,
            "response_format": response_format,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
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


def _stage_for_messages(messages):
    system = messages[0]["content"]
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
    raise AssertionError("unknown history review stage prompt")


def _expected_summary(
    *, correct=0, incorrect=0, uncertain=0, no_claim=0,
):
    factual = correct + incorrect + uncertain
    return (
        f"Reviewed {factual} factual claims: {correct} correct, "
        f"{incorrect} incorrect, {uncertain} uncertain; {no_claim} "
        f"source(s) contained no verifiable factual claim."
    )


def _stage_responder(stage1_response, stage2_response):
    """Return a SpyClient response callable dispatched by stage prompt."""

    def respond(messages):
        if _stage_for_messages(messages) == "stage1":
            return stage1_response(messages)
        return stage2_response(messages)

    return respond


def _valid_response(messages, *, summary=None, coverage_note=None):
    """Stage-aware deterministic response for generic service tests."""

    stage = _stage_for_messages(messages)
    user_data = json.loads(messages[1]["content"])

    if stage == "stage1":
        sources = []
        for source in user_data["sources"]:
            content = source["content"]
            if content.strip():
                quote = content[:256]
                claims = [{
                    "claim_text": quote,
                    "source_quote": quote,
                }]
            else:
                claims = []
            sources.append({
                "source_message_id": source["source_message_id"],
                "claims": claims,
            })
        return json.dumps(
            {"overflow": False, "sources": sources},
            ensure_ascii=False,
        )

    claims = [
        {
            "claim_id": claim["claim_id"],
            "verdict": "correct",
            "correction_text": None,
            "explanation_text": None,
        }
        for claim in user_data["claims"]
    ]
    return json.dumps({"claims": claims}, ensure_ascii=False)


def _mixed_content_response(messages):
    stage = _stage_for_messages(messages)
    user_data = json.loads(messages[1]["content"])

    if stage == "stage1":
        source_ids = [
            source["source_message_id"] for source in user_data["sources"]
        ]
        assert len(source_ids) == 3
        return json.dumps({
            "overflow": False,
            "sources": [
                {
                    "source_message_id": source_ids[0],
                    "claims": [
                        {
                            "claim_text": (
                                "The Pacific Ocean is smaller than the "
                                "Atlantic Ocean."
                            ),
                            "source_quote": (
                                "the Pacific Ocean is smaller than the "
                                "Atlantic Ocean"
                            ),
                        },
                        {
                            "claim_text": (
                                "Mount Everest is located in South America."
                            ),
                            "source_quote": (
                                "Mount Everest is located in South America"
                            ),
                        },
                    ],
                },
                {
                    "source_message_id": source_ids[1],
                    "claims": [
                        {
                            "claim_text": (
                                "Canberra is the capital of Australia."
                            ),
                            "source_quote": (
                                "Canberra is the capital of Australia"
                            ),
                        },
                    ],
                },
                {"source_message_id": source_ids[2], "claims": []},
            ],
        }, ensure_ascii=False)

    return json.dumps({
        "claims": [
            {
                "claim_id": 1,
                "verdict": "incorrect",
                "correction_text": (
                    "The Pacific Ocean is larger than the Atlantic Ocean."
                ),
                "explanation_text": "The first claim is factually incorrect.",
            },
            {
                "claim_id": 2,
                "verdict": "incorrect",
                "correction_text": (
                    "Mount Everest is not located in South America."
                ),
                "explanation_text": "The second claim is factually incorrect.",
            },
            {
                "claim_id": 3,
                "verdict": "correct",
                "correction_text": None,
                "explanation_text": None,
            },
        ],
    }, ensure_ascii=False)


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
    assert result.summary == _expected_summary(correct=1)
    assert result.coverage_note is None
    assert result.error_code is None
    assert result.raw_output_truncated is False
    assert len(result.findings) == 1
    assert result.findings[0].seq == 1
    assert result.findings[0].source_message_id == messages[0].id
    assert result.findings[0].verdict == "correct"
    assert len(api_client.messages) == 2

    stored = _refresh_review(db, review.id)
    assert stored.status == "completed"
    assert stored.summary == _expected_summary(correct=1)
    assert stored.coverage_note is None
    assert stored.error_code is None
    assert stored.started_at is not None
    assert stored.completed_at == stored.updated_at
    findings = _findings_for_review(db, review.id)
    assert [f.seq for f in findings] == [1]
    assert findings[0].source_message_id == messages[0].id
    assert findings[0].verdict == "correct"


@pytest.mark.anyio
async def test_mixed_facts_and_instruction_persist_in_source_order(
    service, db, api_client,
):
    contents = (
        "I’m teaching you two geography claims: the Pacific Ocean is "
        "smaller than the Atlantic Ocean, and Mount Everest is located in "
        "South America. Record both claims without evaluating them.",
        "Add another claim to the lesson: Canberra is the capital of "
        "Australia. Please confirm only what I have taught so far.",
        "Summarize my geography lesson in three numbered points.",
    )
    session, event, messages, review, sources = await _prepare_pending(
        service, db, contents=contents,
    )
    assert review.status == "pending"
    assert review.prompt_version_snapshot == HISTORY_REVIEW_PIPELINE_VERSION

    api_client.response = _mixed_content_response

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "completed"
    assert result.called_llm is True
    assert len(api_client.messages) == 2
    assert [finding.seq for finding in result.findings] == [1, 2, 3, 4]
    assert [finding.source_message_id for finding in result.findings] == [
        messages[0].id,
        messages[0].id,
        messages[1].id,
        messages[2].id,
    ]
    assert [finding.verdict for finding in result.findings] == [
        "incorrect",
        "incorrect",
        "correct",
        "not_a_claim",
    ]

    system_prompt = api_client.messages[0][0]["content"]
    assert "每个独立事实主张单独一条 claim" in system_prompt
    assert "overflow" in system_prompt
    assert "Pacific" not in system_prompt
    assert "Everest" not in system_prompt
    assert "Canberra" not in system_prompt

    stored = _refresh_review(db, review.id)
    assert stored.status == "completed"
    assert stored.prompt_version_snapshot == HISTORY_REVIEW_PIPELINE_VERSION


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
    assert observed == [False, False]


@pytest.mark.anyio
async def test_completed_idempotent_returns_findings_without_llm(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = lambda msgs: _valid_response(msgs)
    first = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    assert first.status == "completed"
    assert len(api_client.messages) == 2

    second = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert second.status == "completed"
    assert second.called_llm is False
    assert second.summary == _expected_summary(correct=1)
    assert second.findings == first.findings
    assert len(api_client.messages) == 2


@pytest.mark.anyio
async def test_old_v1_completed_review_is_returned_unchanged(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    now = utc_now()
    stored = _refresh_review(db, review.id)
    stored.status = "completed"
    stored.summary = "legacy v1 summary"
    stored.coverage_note = "legacy v1 coverage"
    stored.prompt_version_snapshot = "history-review-prompt-v1"
    stored.completed_at = now
    stored.updated_at = now
    db.add(HistoryReviewFinding(
        review_id=review.id,
        seq=1,
        source_message_id=messages[0].id,
        verdict="incorrect",
        claim_text="legacy v1 claim",
        correction_text="legacy v1 correction",
        explanation_text="legacy v1 explanation",
    ))
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "completed"
    assert result.called_llm is False
    assert result.summary == "legacy v1 summary"
    assert result.coverage_note == "legacy v1 coverage"
    assert result.findings[0].claim_text == "legacy v1 claim"
    assert api_client.messages == []

    refreshed = _refresh_review(db, review.id)
    assert refreshed.status == "completed"
    assert refreshed.summary == "legacy v1 summary"
    assert refreshed.prompt_version_snapshot == "history-review-prompt-v1"


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


@pytest.mark.parametrize(
    ("code", "expected_message"),
    [
        (
            "history_review_extraction_json_syntax_error",
            "History review extraction output is not valid JSON.",
        ),
        (
            "history_review_extraction_json_structure_error",
            "History review extraction output has an invalid JSON structure.",
        ),
        (
            "history_review_extraction_json_semantic_error",
            "History review extraction output violates the semantic contract.",
        ),
        (
            "history_review_extraction_source_invalid",
            "History review extraction references an invalid source.",
        ),
        (
            "history_review_extraction_source_uncovered",
            "History review extraction does not cover every source.",
        ),
        (
            "history_review_extraction_quote_invalid",
            "History review extraction contains an invalid source quote.",
        ),
        (
            "history_review_extraction_too_large",
            "History review extraction output exceeds configured limits.",
        ),
        (
            "history_review_extraction_input_too_large",
            "History review extraction input exceeds configured limits.",
        ),
        (
            "history_review_extraction_capacity_exceeded",
            "History review extraction cannot fit all factual claims.",
        ),
        (
            "history_review_verification_json_syntax_error",
            "History review verification output is not valid JSON.",
        ),
        (
            "history_review_verification_json_structure_error",
            "History review verification output has an invalid JSON structure.",
        ),
        (
            "history_review_verification_json_semantic_error",
            "History review verification output violates the semantic contract.",
        ),
        (
            "history_review_verification_claim_invalid",
            "History review verification references an invalid claim.",
        ),
        (
            "history_review_verification_claim_uncovered",
            "History review verification does not cover every claim.",
        ),
        (
            "history_review_verification_too_large",
            "History review verification output exceeds configured limits.",
        ),
        (
            "history_review_verification_input_too_large",
            "History review verification input exceeds configured limits.",
        ),
    ],
)
def test_two_stage_pipeline_error_codes_have_safe_messages(
    code, expected_message,
):
    actual = execution_module._safe_error_message(code)
    assert actual == expected_message
    assert actual != "History review execution failed."


def test_unknown_pipeline_error_code_uses_safe_fallback():
    assert execution_module._safe_error_message(
        "history_review_unmapped_error_code"
    ) == "History review execution failed."


@pytest.mark.parametrize(
    ("code", "expected_message"),
    [
        (
            "history_review_output_too_large",
            "History review output exceeds configured limits.",
        ),
        (
            "history_review_json_syntax_error",
            "History review output is not valid JSON.",
        ),
        (
            "history_review_json_structure_error",
            "History review output has an invalid JSON structure.",
        ),
        (
            "history_review_json_semantic_error",
            "History review output violates the semantic contract.",
        ),
        (
            "history_review_source_reference_invalid",
            "History review output references an invalid source.",
        ),
        (
            "history_review_source_uncovered",
            "History review output does not cover every source.",
        ),
    ],
)
def test_legacy_single_stage_error_messages_are_preserved(
    code, expected_message,
):
    assert execution_module._safe_error_message(code) == expected_message


def test_history_review_stages_docstring_reflects_production_wiring():
    import backend.history_review_stages as stages_module

    doc = stages_module.__doc__ or ""
    assert (
        "The production execution path still uses the single-stage v2 prompt."
        not in doc
    )
    assert (
        "Phase 2B will wire these pure functions into the service layer."
        not in doc
    )


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

    expected_code = "history_review_extraction_json_syntax_error"
    expected_message = execution_module._safe_error_message(expected_code)
    assert result.status == "failed"
    assert result.error_code == expected_code
    assert result.error_message == expected_message
    assert len(api_client.messages) == 2
    stored = _assert_running_failure_shape(db, review.id)
    assert stored.error_code == expected_code
    assert stored.error_message == expected_message
    assert stored.raw_output is not None
    assert "not json" in stored.raw_output


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
    assert result.error_code == (
        "history_review_extraction_json_structure_error"
    )

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
        "overflow": True,
        "sources": [
            {"source_message_id": messages[0].id, "claims": []},
        ],
    })
    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    assert result.error_code == (
        "history_review_extraction_json_semantic_error"
    )


@pytest.mark.anyio
async def test_parser_reference_and_uncovered_codes(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.response = _stage_responder(
        _valid_response,
        lambda msgs: json.dumps({
            "claims": [{
                "claim_id": 999999,
                "verdict": "correct",
                "correction_text": None,
                "explanation_text": None,
            }],
        }),
    )
    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    expected_code = "history_review_verification_claim_invalid"
    expected_message = execution_module._safe_error_message(expected_code)
    assert result.error_code == expected_code
    assert result.error_message == expected_message
    stored = _assert_running_failure_shape(db, review.id)
    assert stored.error_code == expected_code
    assert stored.error_message == expected_message


@pytest.mark.anyio
async def test_parser_uncovered_code_for_two_sources(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db, contents=("one", "two"),
    )
    api_client.response = _stage_responder(
        _valid_response,
        lambda msgs: json.dumps({
            "claims": [{
                "claim_id": 1,
                "verdict": "correct",
                "correction_text": None,
                "explanation_text": None,
            }],
        }),
    )

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.error_code == (
        "history_review_verification_claim_uncovered"
    )
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
    assert result.error_code == "history_review_extraction_too_large"
    assert result.called_llm is True
    assert result.raw_output_truncated is True
    stored = _assert_running_failure_shape(db, review.id)

    expected_audit = json.dumps(
        {
            "attempts": [
                {
                    "attempt": 1,
                    "raw_output": huge,
                    "stage": "stage1",
                },
            ],
            "pipeline_version": "history-review-pipeline-v1",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert len(expected_audit) > 16000
    assert stored.raw_output == expected_audit[:16000]
    assert stored.raw_output_truncated is True
    assert len(api_client.messages) == 1


@pytest.mark.anyio
async def test_findings_keep_stage1_source_and_claim_order(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db, contents=("one", "two"),
    )
    api_client.response = _stage_responder(
        lambda msgs: json.dumps({
            "overflow": False,
            "sources": [
                {
                    "source_message_id": messages[0].id,
                    "claims": [
                        {
                            "claim_text": "first claim",
                            "source_quote": "one",
                        },
                        {
                            "claim_text": "second claim",
                            "source_quote": "one",
                        },
                    ],
                },
                {
                    "source_message_id": messages[1].id,
                    "claims": [
                        {
                            "claim_text": "third claim",
                            "source_quote": "two",
                        },
                    ],
                },
            ],
        }),
        lambda msgs: json.dumps({
            "claims": [
                {
                    "claim_id": 1,
                    "verdict": "correct",
                    "correction_text": None,
                    "explanation_text": None,
                },
                {
                    "claim_id": 2,
                    "verdict": "uncertain",
                    "correction_text": None,
                    "explanation_text": "not sure",
                },
                {
                    "claim_id": 3,
                    "verdict": "incorrect",
                    "correction_text": "fix",
                    "explanation_text": "why",
                },
            ],
        }),
    )

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "completed"
    assert len(api_client.messages) == 2
    assert [f.seq for f in result.findings] == [1, 2, 3]
    assert [f.source_message_id for f in result.findings] == [
        messages[0].id,
        messages[0].id,
        messages[1].id,
    ]
    assert [f.claim_text for f in result.findings] == [
        "first claim",
        "second claim",
        "third claim",
    ]
    assert [f.verdict for f in result.findings] == [
        "correct",
        "uncertain",
        "incorrect",
    ]
    rows = _findings_for_review(db, review.id)
    assert [row.seq for row in rows] == [1, 2, 3]
    assert [row.source_message_id for row in rows] == [
        messages[0].id,
        messages[0].id,
        messages[1].id,
    ]
    assert [row.claim_text for row in rows] == [
        "first claim",
        "second claim",
        "third claim",
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
    assert len(api_client.messages) == 2
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
    assert len(api_client.messages) == 2


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
    assert len(api_client.messages) == 2

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

    async def _fail_pipeline(*, client, sources):
        raise HistoryReviewPipelineError(
            code="history_review_extraction_input_too_large",
            audit_raw_output=None,
            called_llm=False,
            llm_call_count=0,
        )

    monkeypatch.setattr(
        execution_module,
        "run_history_review_pipeline",
        _fail_pipeline,
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


@pytest.mark.anyio
async def test_zero_claim_completes_without_stage2_and_stage1_only_audit(
    service, db, api_client,
):
    contents = ("just a menu", "please summarize")
    session, event, messages, review, sources = await _prepare_pending(
        service, db, contents=contents,
    )

    def respond(msgs):
        assert _stage_for_messages(msgs) == "stage1"
        user_data = json.loads(msgs[1]["content"])
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

    api_client.response = respond

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "completed"
    assert result.called_llm is True
    assert len(api_client.messages) == 1
    assert [f.seq for f in result.findings] == [1, 2]
    assert [f.source_message_id for f in result.findings] == [
        messages[0].id,
        messages[1].id,
    ]
    assert [f.verdict for f in result.findings] == [
        "not_a_claim",
        "not_a_claim",
    ]
    assert [f.claim_text for f in result.findings] == [
        contents[0],
        contents[1],
    ]

    stored = _refresh_review(db, review.id)
    assert stored.status == "completed"
    audit = json.loads(stored.raw_output)
    assert audit["pipeline_version"] == HISTORY_REVIEW_PIPELINE_VERSION
    assert [
        attempt["stage"] for attempt in audit["attempts"]
    ] == ["stage1"]
    assert [row.verdict for row in _findings_for_review(
        db, review.id
    )] == ["not_a_claim", "not_a_claim"]


@pytest.mark.anyio
async def test_stage1_repair_success_persists_audit_and_attempt_count(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db, contents=("one",),
    )
    stage1_calls = []
    transaction_states = []
    api_client.before_generate = lambda: transaction_states.append(
        db.in_transaction()
    )

    def respond(msgs):
        stage = _stage_for_messages(msgs)
        if stage == "stage1":
            stage1_calls.append(msgs)
            if len(stage1_calls) == 1:
                return "not json"
            user_data = json.loads(msgs[1]["content"])
            source = user_data["sources"][0]
            return json.dumps({
                "overflow": False,
                "sources": [{
                    "source_message_id": source["source_message_id"],
                    "claims": [{
                        "claim_text": "first claim",
                        "source_quote": "one",
                    }],
                }],
            }, ensure_ascii=False)

        user_data = json.loads(msgs[1]["content"])
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

    api_client.response = respond

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "completed"
    assert result.called_llm is True
    assert len(api_client.messages) == 3
    assert len(stage1_calls) == 2
    assert stage1_calls[1][0]["content"].startswith(
        HISTORY_REVIEW_STAGE_1_SYSTEM_PROMPT + "\n"
    )
    assert transaction_states == [False, False, False]

    stored = _refresh_review(db, review.id)
    assert stored.status == "completed"
    assert stored.attempt_count == 1
    audit = json.loads(stored.raw_output)
    assert [
        (attempt["stage"], attempt["attempt"])
        for attempt in audit["attempts"]
    ] == [("stage1", 1), ("stage1", 2), ("stage2", 1)]


@pytest.mark.anyio
async def test_pipeline_generate_options_use_frozen_stage_settings(
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

    assert result.status == "completed"
    assert len(api_client.calls) == 2
    assert [
        _stage_for_messages(call["messages"])
        for call in api_client.calls
    ] == ["stage1", "stage2"]
    assert [call["response_format"] for call in api_client.calls] == [
        {"type": "json_object"},
        {"type": "json_object"},
    ]
    assert [call["temperature"] for call in api_client.calls] == [0, 0]
    assert [call["max_tokens"] for call in api_client.calls] == [
        STAGE1_RESERVED_OUTPUT_TOKENS,
        STAGE2_RESERVED_OUTPUT_TOKENS,
    ]


@pytest.mark.anyio
async def test_pipeline_error_audit_on_repeated_retryable_contract_error(
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
    assert result.called_llm is True
    assert result.error_code == (
        "history_review_extraction_json_syntax_error"
    )
    assert result.findings == ()
    assert len(api_client.messages) == 2

    stored = _assert_running_failure_shape(db, review.id)
    audit = json.loads(stored.raw_output)
    assert audit["pipeline_version"] == HISTORY_REVIEW_PIPELINE_VERSION
    assert [
        (attempt["stage"], attempt["attempt"], attempt["raw_output"])
        for attempt in audit["attempts"]
    ] == [
        ("stage1", 1, "not json"),
        ("stage1", 2, "not json"),
    ]
    assert _findings_for_review(db, review.id) == []


@pytest.mark.anyio
async def test_cancelled_error_marks_interrupted_and_reraises(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await service.execute_history_review(
            session_id=session.id,
            review_id=review.id,
            db=db,
        )

    assert len(api_client.messages) == 1
    stored = _refresh_review(db, review.id)
    assert stored.status == "failed"
    assert stored.error_code == HISTORY_REVIEW_EXECUTION_INTERRUPTED
    assert stored.error_message == execution_module._safe_error_message(
        HISTORY_REVIEW_EXECUTION_INTERRUPTED
    )
    assert stored.started_at is not None
    assert stored.completed_at is not None
    assert stored.completed_at == stored.updated_at
    assert stored.summary is None
    assert stored.coverage_note is None
    assert stored.raw_output is None
    assert stored.raw_output_truncated is False
    assert _findings_for_review(db, review.id) == []

    second = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )
    assert second.status == "failed"
    assert second.called_llm is False
    assert second.error_code == HISTORY_REVIEW_EXECUTION_INTERRUPTED
    assert len(api_client.messages) == 1


@pytest.mark.anyio
async def test_pending_legacy_prompt_v2_is_unsupported_without_llm(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored = _refresh_review(db, review.id)
    stored.prompt_version_snapshot = "history-review-prompt-v2"
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == "failed"
    assert result.called_llm is False
    assert result.error_code == PROMPT_VERSION_UNSUPPORTED
    assert result.findings == ()
    assert api_client.messages == []

    stored = _refresh_review(db, review.id)
    assert stored.status == "failed"
    assert stored.prompt_version_snapshot == "history-review-prompt-v2"
    assert stored.error_code == PROMPT_VERSION_UNSUPPORTED
    assert _findings_for_review(db, review.id) == []


@pytest.mark.parametrize("terminal_status", ["completed", "failed"])
@pytest.mark.anyio
async def test_terminal_legacy_prompt_v2_is_returned_without_reexecution(
    service, db, api_client, terminal_status,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    now = utc_now()
    stored = _refresh_review(db, review.id)
    stored.status = terminal_status
    stored.prompt_version_snapshot = "history-review-prompt-v2"
    stored.completed_at = now
    stored.updated_at = now
    if terminal_status == "completed":
        stored.summary = "legacy v2 summary"
    else:
        stored.error_code = REVIEWER_UNAVAILABLE
        stored.error_message = "Reviewer profile is unavailable."
    db.commit()

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
    )

    assert result.status == terminal_status
    assert result.called_llm is False
    assert api_client.messages == []

    refreshed = _refresh_review(db, review.id)
    assert refreshed.status == terminal_status
    assert refreshed.prompt_version_snapshot == "history-review-prompt-v2"


async def _make_interrupted_review(
    service,
    db,
    *,
    error_code=HISTORY_REVIEW_EXECUTION_INTERRUPTED,
    attempt_count=1,
    with_finding=False,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored = _refresh_review(db, review.id)
    now = utc_now()
    stored.status = "failed"
    stored.error_code = error_code
    stored.error_message = execution_module._safe_error_message(error_code)
    stored.summary = "old summary"
    stored.coverage_note = "old coverage"
    stored.raw_output = "old audit"
    stored.raw_output_truncated = True
    stored.attempt_count = attempt_count
    stored.started_at = now
    stored.completed_at = now
    stored.updated_at = now
    if with_finding:
        db.add(HistoryReviewFinding(
            review_id=review.id,
            seq=1,
            source_message_id=sources[0].message_id,
            verdict="correct",
            claim_text="old claim",
            correction_text=None,
            explanation_text=None,
        ))
    db.commit()
    return session, event, messages, stored, sources, review


@pytest.mark.anyio
async def test_cancelled_cleanup_failure_does_not_mask_cancellation(
    service, db, api_client, monkeypatch,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    api_client.error = asyncio.CancelledError()

    def _fail_cleanup(*args, **kwargs):
        raise SQLAlchemyError("simulated cancellation cleanup failure")

    monkeypatch.setattr(service, "_record_running_failure", _fail_cleanup)

    with pytest.raises(asyncio.CancelledError):
        await service.execute_history_review(
            session_id=session.id,
            review_id=review.id,
            db=db,
        )

    stored = _refresh_review(db, review.id)
    assert stored.status == "running"


def test_recover_abandoned_running_reviews_noop(service, db):
    assert service.recover_abandoned_running_reviews(db) == 0


@pytest.mark.anyio
async def test_recover_abandoned_running_reviews_marks_interrupted(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored = _refresh_review(db, review.id)
    stored.status = "running"
    stored.started_at = utc_now()
    db.add(HistoryReviewFinding(
        review_id=review.id,
        seq=1,
        source_message_id=sources[0].message_id,
        verdict="correct",
        claim_text="stale finding",
        correction_text=None,
        explanation_text=None,
    ))
    db.commit()

    recovered = service.recover_abandoned_running_reviews(db)

    assert recovered == 1
    stored = _refresh_review(db, review.id)
    assert stored.status == "failed"
    assert stored.error_code == HISTORY_REVIEW_EXECUTION_INTERRUPTED
    assert stored.error_message == execution_module._safe_error_message(
        HISTORY_REVIEW_EXECUTION_INTERRUPTED
    )
    assert stored.summary is None
    assert stored.coverage_note is None
    assert stored.raw_output is None
    assert stored.raw_output_truncated is False
    assert stored.started_at is not None
    assert stored.completed_at is not None
    assert _findings_for_review(db, review.id) == []
    assert api_client.messages == []

    assert service.recover_abandoned_running_reviews(db) == 0


@pytest.mark.anyio
async def test_recover_abandoned_running_reviews_failure_raises(
    service, db, engine,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored = _refresh_review(db, review.id)
    stored.status = "running"
    stored.started_at = utc_now()
    db.commit()

    with engine.begin() as conn:
        conn.execute(sa_text(
            "CREATE TRIGGER fail_running_recovery "
            "BEFORE UPDATE ON history_reviews "
            "WHEN OLD.status = 'running' AND NEW.status = 'failed' "
            "BEGIN SELECT RAISE(FAIL, 'simulated recovery failure'); END"
        ))
    try:
        with pytest.raises(SQLAlchemyError, match="simulated recovery failure"):
            service.recover_abandoned_running_reviews(db)
    finally:
        with engine.begin() as conn:
            conn.execute(sa_text(
                "DROP TRIGGER IF EXISTS fail_running_recovery"
            ))


@pytest.mark.anyio
async def test_retry_non_interrupted_failed_is_rejected(
    service, db, api_client,
):
    session, event, messages, stored, sources, review = (
        await _make_interrupted_review(
            service,
            db,
            error_code=LLM_UPSTREAM_ERROR,
        )
    )

    with pytest.raises(HistoryReviewRetryNotAllowed):
        await service.execute_history_review(
            session_id=session.id,
            review_id=review.id,
            db=db,
            retry_failed=True,
        )

    assert api_client.messages == []
    refreshed = _refresh_review(db, review.id)
    assert refreshed.status == "failed"
    assert refreshed.error_code == LLM_UPSTREAM_ERROR
    assert refreshed.attempt_count == 1


@pytest.mark.anyio
async def test_retry_interrupted_reuses_review_and_sources(
    service, db, api_client,
):
    session, event, messages, stored, sources, review = (
        await _make_interrupted_review(
            service,
            db,
            with_finding=True,
        )
    )
    source_ids = [source.id for source in sources]
    snapshot = (
        stored.reviewer_llm_profile_id_snapshot,
        stored.reviewer_llm_profile_kind_snapshot,
        stored.reviewer_llm_model_snapshot,
        stored.prompt_version_snapshot,
        stored.budget_version_snapshot,
        stored.selection_policy_version,
        stored.remote_history_acknowledged,
        stored.remote_history_ack_message_count,
    )
    api_client.response = lambda msgs: _valid_response(msgs)

    result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
        retry_failed=True,
    )

    assert result.status == "completed"
    assert len(api_client.messages) == 2
    refreshed = _refresh_review(db, review.id)
    assert refreshed.id == review.id
    assert refreshed.status == "completed"
    assert refreshed.attempt_count == 2
    assert refreshed.error_code is None
    assert refreshed.error_message is None
    assert refreshed.summary == _expected_summary(correct=1)
    assert refreshed.raw_output is not None
    assert refreshed.raw_output != "old audit"
    assert [
        source.id for source in _refresh_sources(db, review.id)
    ] == source_ids
    findings = _findings_for_review(db, review.id)
    assert len(findings) == 1
    assert findings[0].claim_text == "claim"
    assert (
        refreshed.reviewer_llm_profile_id_snapshot,
        refreshed.reviewer_llm_profile_kind_snapshot,
        refreshed.reviewer_llm_model_snapshot,
        refreshed.prompt_version_snapshot,
        refreshed.budget_version_snapshot,
        refreshed.selection_policy_version,
        refreshed.remote_history_acknowledged,
        refreshed.remote_history_ack_message_count,
    ) == snapshot


@pytest.mark.anyio
async def test_retry_completed_and_running_are_authoritative_noops(
    service, db, api_client,
):
    session, event, messages, review, sources = await _prepare_pending(
        service, db,
    )
    stored = _refresh_review(db, review.id)
    now = utc_now()
    stored.status = "completed"
    stored.summary = "done"
    stored.completed_at = now
    stored.updated_at = now
    db.commit()

    completed_result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
        retry_failed=True,
    )
    assert completed_result.status == "completed"
    assert completed_result.called_llm is False
    assert api_client.messages == []

    stored = _refresh_review(db, review.id)
    stored.status = "running"
    stored.started_at = now
    stored.completed_at = None
    stored.updated_at = now
    db.commit()

    running_result = await service.execute_history_review(
        session_id=session.id,
        review_id=review.id,
        db=db,
        retry_failed=True,
    )
    assert running_result.status == "running"
    assert running_result.called_llm is False
    assert api_client.messages == []


@pytest.mark.anyio
async def test_retry_cancelled_again_marks_interrupted(
    service, db, api_client,
):
    session, event, messages, stored, sources, review = (
        await _make_interrupted_review(service, db)
    )
    api_client.error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await service.execute_history_review(
            session_id=session.id,
            review_id=review.id,
            db=db,
            retry_failed=True,
        )

    refreshed = _refresh_review(db, review.id)
    assert refreshed.status == "failed"
    assert refreshed.error_code == HISTORY_REVIEW_EXECUTION_INTERRUPTED
    assert refreshed.attempt_count == 2
    assert refreshed.summary is None
    assert refreshed.raw_output is None
    assert _findings_for_review(db, review.id) == []


@pytest.mark.anyio
async def test_concurrent_retry_only_one_execution(
    service, db, session_factory, api_client,
):
    session, event, messages, stored, sources, review = (
        await _make_interrupted_review(service, db)
    )
    api_client.response = lambda msgs: _valid_response(msgs)
    db_one = session_factory()
    db_two = session_factory()
    try:
        results = await asyncio.gather(
            service.execute_history_review(
                session_id=session.id,
                review_id=review.id,
                db=db_one,
                retry_failed=True,
            ),
            service.execute_history_review(
                session_id=session.id,
                review_id=review.id,
                db=db_two,
                retry_failed=True,
            ),
        )
    finally:
        db_one.close()
        db_two.close()

    assert [result.status for result in results].count("completed") >= 1
    assert len(api_client.messages) == 2
    refreshed = _refresh_review(db, review.id)
    assert refreshed.status == "completed"
    assert refreshed.attempt_count == 2
    assert len(_findings_for_review(db, review.id)) == 1
    assert len(_refresh_sources(db, review.id)) == len(sources)


@pytest.mark.anyio
async def test_retry_reset_failure_rolls_back(
    service, db, engine, api_client,
):
    session, event, messages, stored, sources, review = (
        await _make_interrupted_review(
            service,
            db,
            with_finding=True,
        )
    )
    with engine.begin() as conn:
        conn.execute(sa_text(
            "CREATE TRIGGER fail_retry_finding_delete "
            "BEFORE DELETE ON history_review_findings "
            "BEGIN SELECT RAISE(FAIL, 'simulated retry reset failure'); END"
        ))
    try:
        with pytest.raises(SQLAlchemyError, match="simulated retry reset failure"):
            await service.execute_history_review(
                session_id=session.id,
                review_id=review.id,
                db=db,
                retry_failed=True,
            )
    finally:
        with engine.begin() as conn:
            conn.execute(sa_text(
                "DROP TRIGGER IF EXISTS fail_retry_finding_delete"
            ))

    refreshed = _refresh_review(db, review.id)
    assert refreshed.status == "failed"
    assert refreshed.error_code == HISTORY_REVIEW_EXECUTION_INTERRUPTED
    assert refreshed.attempt_count == 1
    assert refreshed.summary == "old summary"
    assert refreshed.coverage_note == "old coverage"
    assert refreshed.raw_output == "old audit"
    findings = _findings_for_review(db, review.id)
    assert len(findings) == 1
    assert findings[0].claim_text == "old claim"
    assert api_client.messages == []
