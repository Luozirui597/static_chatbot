"""Iteration 2C history review preparation service tests."""

import asyncio
import copy
from contextlib import asynccontextmanager
from datetime import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.chat_service import ChatService, SessionLockRegistry
from backend.database import create_database_engine, create_tables, run_migrations
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
from backend.history_review_selection import (
    HISTORY_REVIEW_BUDGET_VERSION,
    HISTORY_REVIEW_SELECTION_POLICY_VERSION,
    HistoryReviewSourceMessageTooLarge,
)
from backend.history_review_prompt import HISTORY_REVIEW_PROMPT_VERSION
from backend.history_review_service import (
    HistoryReviewPreparationResult,
    HistoryReviewService,
)
from backend.interaction_modes import CORRECTIVE_MODE, RECEIVE_TEACHING_MODE
from backend.llm_profiles import LLMProfile, LLMProfileRegistry
from backend.models import (
    ChatSession,
    HistoryReview,
    HistoryReviewSource,
    Message,
    ModeSwitchEvent,
)

FAKE_MODEL = "fake"
LOCAL_MODEL = "local-model"
API_MODEL = "api-model"


class SpyClient:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages):
        self.calls += 1
        return "unused"


class CountingLockRegistry(SessionLockRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.entered: list[int] = []

    @asynccontextmanager
    async def session_lock(self, session_id):
        self.entered.append(session_id)
        async with super().session_lock(session_id):
            yield


@pytest.fixture
def engine(tmp_path):
    eng = create_database_engine(f"sqlite:///{tmp_path / 'service.db'}")
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
def api_client():
    return SpyClient()


@pytest.fixture
def profiles(api_client):
    return LLMProfileRegistry([
        LLMProfile(
            id="default", label="Fake", kind="fake", model=FAKE_MODEL,
            client=api_client, is_default=True,
        ),
        LLMProfile(
            id="local", label="Local", kind="local", model=LOCAL_MODEL,
            client=api_client,
        ),
        LLMProfile(
            id="api", label="API", kind="api", model=API_MODEL,
            client=api_client,
        ),
    ])


@pytest.fixture
def lock_registry():
    return SessionLockRegistry()


@pytest.fixture
def service(profiles, lock_registry):
    return HistoryReviewService(
        profiles=profiles, lock_registry=lock_registry,
    )


def _make_session(
    db_session,
    profile_id="default",
    model_snapshot=FAKE_MODEL,
) -> ChatSession:
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
    role="user",
    interaction_mode=RECEIVE_TEACHING_MODE,
    profile_id="default",
    kind="fake",
    model_snapshot=FAKE_MODEL,
):
    message = Message(
        session_id=session.id,
        role=role,
        content=content,
        interaction_mode_snapshot=interaction_mode,
        llm_profile_id_snapshot=profile_id,
        llm_profile_kind_snapshot=kind,
        llm_model_snapshot=model_snapshot,
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(message)
    return message


def _make_event(
    db_session,
    session,
    *,
    from_mode=RECEIVE_TEACHING_MODE,
    to_mode=CORRECTIVE_MODE,
    version=HISTORY_BOUNDARY_VERSION,
    through=None,
    count=None,
):
    event = ModeSwitchEvent(
        session_id=session.id,
        from_mode=from_mode,
        to_mode=to_mode,
        history_boundary_version=version,
        history_through_message_id=through,
        reviewable_user_message_count=count,
    )
    db_session.add(event)
    db_session.commit()
    db_session.refresh(event)
    return event


class TestPrepareInputAndEvent:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("session_id", "event_id", "ack"),
        [
            (True, 1, False),
            (0, 1, False),
            (-1, 1, False),
            ("1", 1, False),
            (1, True, False),
            (1, 0, False),
            (1, -1, False),
            (1, 1, 1),
            (1, 1, "false"),
        ],
    )
    async def test_invalid_inputs_are_rejected_before_db_access(
        self, service, db, session_id, event_id, ack,
    ):
        with pytest.raises(ValueError):
            await service.prepare_history_review(
                session_id=session_id,
                mode_switch_event_id=event_id,
                acknowledge_remote_history=ack,
                db=db,
            )

    @pytest.mark.anyio
    async def test_session_not_found(self, service, db):
        with pytest.raises(HistoryReviewSessionNotFound):
            await service.prepare_history_review(
                session_id=1,
                mode_switch_event_id=1,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_event_not_found_and_other_session_same_semantics(
        self, service, db,
    ):
        session_a = _make_session(db, "default", FAKE_MODEL)
        session_b = _make_session(db, "default", FAKE_MODEL)
        event_b = _make_event(
            db, session_b, through=None, count=0,
        )

        with pytest.raises(HistoryReviewEventNotFound):
            await service.prepare_history_review(
                session_id=session_a.id,
                mode_switch_event_id=event_b.id,
                acknowledge_remote_history=False,
                db=db,
            )

        with pytest.raises(HistoryReviewEventNotFound):
            await service.prepare_history_review(
                session_id=session_a.id,
                mode_switch_event_id=999999,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_event_direction_must_be_receive_to_corrective(
        self, service, db,
    ):
        session = _make_session(db)
        wrong_direction = _make_event(
            db, session,
            from_mode=CORRECTIVE_MODE,
            to_mode=RECEIVE_TEACHING_MODE,
            through=1,
            count=1,
        )
        with pytest.raises(HistoryReviewEventNotReviewable):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=wrong_direction.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "version",
        [None, "history-boundary-v0", "bad"],
    )
    async def test_target_boundary_version_must_be_v1(
        self, service, db, version,
    ):
        session = _make_session(db)
        event = _make_event(
            db, session, version=version, through=1, count=1,
        )
        with pytest.raises(HistoryReviewBoundaryUnavailable):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_target_count_must_not_be_none(self, service, db):
        session = _make_session(db)
        event = _make_event(
            db, session, version=HISTORY_BOUNDARY_VERSION,
            through=1, count=None,
        )
        with pytest.raises(HistoryReviewBoundaryUnavailable):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_upper_bound_none_is_no_reviewable_history(
        self, service, db,
    ):
        session = _make_session(db)
        event = _make_event(
            db, session, version=HISTORY_BOUNDARY_VERSION,
            through=None, count=0,
        )
        with pytest.raises(HistoryReviewNoReviewableHistory):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_legacy_lower_event_is_boundary_unavailable(
        self, service, db,
    ):
        session = _make_session(db)
        _make_event(
            db, session,
            from_mode=CORRECTIVE_MODE,
            to_mode=RECEIVE_TEACHING_MODE,
            version=None,
            through=None,
            count=None,
        )
        event = _make_event(
            db, session, version=HISTORY_BOUNDARY_VERSION,
            through=1, count=1,
        )
        with pytest.raises(HistoryReviewBoundaryUnavailable):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_lower_greater_than_upper_is_boundary_unavailable(
        self, service, db,
    ):
        session = _make_session(db)
        _make_event(
            db, session,
            from_mode=CORRECTIVE_MODE,
            to_mode=RECEIVE_TEACHING_MODE,
            version=HISTORY_BOUNDARY_VERSION,
            through=10,
            count=0,
        )
        event = _make_event(
            db, session, version=HISTORY_BOUNDARY_VERSION,
            through=5, count=1,
        )
        with pytest.raises(HistoryReviewBoundaryUnavailable):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_lower_event_chosen_by_id_not_created_at(
        self, service, db,
    ):
        session = _make_session(db)
        for index in range(3):
            _make_message(
                db, session, f"context-{index}",
                role="assistant",
                interaction_mode=RECEIVE_TEACHING_MODE,
            )
        first = _make_event(
            db, session,
            from_mode=CORRECTIVE_MODE,
            to_mode=RECEIVE_TEACHING_MODE,
            version=HISTORY_BOUNDARY_VERSION,
            through=100,
            count=0,
        )
        second = _make_event(
            db, session,
            from_mode=CORRECTIVE_MODE,
            to_mode=RECEIVE_TEACHING_MODE,
            version=HISTORY_BOUNDARY_VERSION,
            through=3,
            count=0,
        )
        first.created_at = datetime(2030, 1, 1)
        second.created_at = datetime(2020, 1, 1)
        db.commit()

        target_message = _make_message(db, session, "latest")
        event = _make_event(
            db, session, version=HISTORY_BOUNDARY_VERSION,
            through=target_message.id, count=1,
        )

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=False,
            db=db,
        )

        assert result.created is True
        assert result.review.lower_bound_kind == "corrective_switch"
        assert result.review.lower_bound_message_id == 3
        assert first.id < second.id < event.id
        assert result.sources[0].message_id == target_message.id

    @pytest.mark.anyio
    async def test_same_mode_event_is_not_used_as_lower(self, service, db):
        session = _make_session(db)
        _make_event(
            db, session,
            from_mode=RECEIVE_TEACHING_MODE,
            to_mode=RECEIVE_TEACHING_MODE,
            version=HISTORY_BOUNDARY_VERSION,
            through=100,
            count=None,
        )
        message = _make_message(db, session, "first cycle")
        event = _make_event(
            db, session, version=HISTORY_BOUNDARY_VERSION,
            through=message.id, count=1,
        )

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=False,
            db=db,
        )
        assert result.review.lower_bound_kind == "session_start"
        assert result.review.lower_bound_message_id is None


class TestEligibleAndSelection:
    @pytest.mark.anyio
    async def test_filters_messages_and_persists_ordered_sources(
        self, service, db,
    ):
        session = _make_session(db)
        first = _make_message(db, session, "claim one")
        _make_message(
            db, session, "assistant reply", role="assistant",
        )
        _make_message(
            db, session, "corrective user",
            interaction_mode=CORRECTIVE_MODE,
        )
        _make_message(
            db, session, "legacy unknown",
            interaction_mode=None,
        )
        second = _make_message(db, session, "claim two")

        event = _make_event(
            db, session,
            version=HISTORY_BOUNDARY_VERSION,
            through=second.id,
            count=2,
        )
        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=False,
            db=db,
        )

        assert result.created is True
        assert result.review.status == "pending"
        assert result.review.eligible_message_count == 2
        assert result.review.eligible_char_count == 18
        assert [source.seq for source in result.sources] == [1, 2]
        assert [source.message_id for source in result.sources] == [
            first.id, second.id,
        ]
        assert [source.content_char_count for source in result.sources] == [
            len(first.content), len(second.content),
        ]

    @pytest.mark.anyio
    async def test_upper_inclusive_lower_exclusive_across_cycles(
        self, service, db,
    ):
        session = _make_session(db)
        first_cycle = _make_message(db, session, "cycle one")
        first_event = _make_event(
            db, session, through=first_cycle.id, count=1,
        )
        first_result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=first_event.id,
            acknowledge_remote_history=False,
            db=db,
        )
        assert [s.message_id for s in first_result.sources] == [
            first_cycle.id
        ]

        resume_event = _make_event(
            db, session,
            from_mode=CORRECTIVE_MODE,
            to_mode=RECEIVE_TEACHING_MODE,
            through=first_cycle.id,
            count=None,
        )
        second_cycle = _make_message(db, session, "cycle two")
        _make_message(
            db, session, "corrective after resume",
            interaction_mode=CORRECTIVE_MODE,
        )
        second_event = _make_event(
            db, session,
            through=second_cycle.id + 1,
            count=1,
        )

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=second_event.id,
            acknowledge_remote_history=False,
            db=db,
        )
        assert result.review.lower_bound_kind == "corrective_switch"
        assert result.review.lower_bound_message_id == first_cycle.id
        assert [s.message_id for s in result.sources] == [second_cycle.id]
        assert resume_event.id < second_event.id

    @pytest.mark.anyio
    async def test_eligible_count_mismatch_is_boundary_unavailable(
        self, service, db,
    ):
        session = _make_session(db)
        _make_message(db, session, "only one")
        event = _make_event(
            db, session, through=1, count=2,
        )
        with pytest.raises(HistoryReviewBoundaryUnavailable):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_no_eligible_messages_is_no_reviewable_history(
        self, service, db,
    ):
        session = _make_session(db)
        assistant = _make_message(
            db, session, "assistant only", role="assistant",
        )
        event = _make_event(
            db, session, through=assistant.id, count=0,
        )
        with pytest.raises(HistoryReviewNoReviewableHistory):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_too_large_source_creates_no_rows(
        self, service, db,
    ):
        session = _make_session(db)
        message = _make_message(db, session, "x" * 4001)
        event = _make_event(
            db, session, through=message.id, count=1,
        )
        with pytest.raises(HistoryReviewSourceMessageTooLarge):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

        assert db.execute(
            select(func.count()).select_from(HistoryReview)
        ).scalar() == 0
        assert db.execute(
            select(func.count()).select_from(HistoryReviewSource)
        ).scalar() == 0


class TestReviewerProfile:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("profile_id", "kind", "model"),
        [
            ("default", "fake", FAKE_MODEL),
            ("local", "local", LOCAL_MODEL),
            ("api", "api", API_MODEL),
        ],
    )
    async def test_reviewer_is_current_session_binding(
        self, service, db, api_client, profile_id, kind, model,
    ):
        session = _make_session(db, profile_id, model)
        message = _make_message(
            db, session, "teaching",
            profile_id=profile_id, kind=kind, model_snapshot=model,
        )
        event = _make_event(
            db, session, through=message.id, count=1,
        )

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=False,
            db=db,
        )

        assert result.review.reviewer_llm_profile_id_snapshot == profile_id
        assert result.review.reviewer_llm_profile_kind_snapshot == kind
        assert result.review.reviewer_llm_model_snapshot == model
        assert result.review.prompt_version_snapshot == (
            HISTORY_REVIEW_PROMPT_VERSION
        )
        assert result.review.budget_version_snapshot == (
            HISTORY_REVIEW_BUDGET_VERSION
        )
        assert result.review.selection_policy_version == (
            HISTORY_REVIEW_SELECTION_POLICY_VERSION
        )
        assert api_client.calls == 0

    @pytest.mark.anyio
    async def test_profile_unavailable_has_no_fallback(
        self, service, db,
    ):
        session = _make_session(db, "missing-profile", "some-model")
        message = _make_message(
            db, session, "teaching",
            profile_id="missing-profile", kind="api",
            model_snapshot="some-model",
        )
        event = _make_event(
            db, session, through=message.id, count=1,
        )
        with pytest.raises(SessionProfileUnavailableError):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_model_changed_is_profile_conflict(self, service, db):
        session = _make_session(db, "default", "changed-model")
        message = _make_message(db, session, "teaching")
        event = _make_event(
            db, session, through=message.id, count=1,
        )
        with pytest.raises(SessionProfileConflictError):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_legacy_unknown_is_profile_conflict(self, service, db):
        session = _make_session(db, "default", None)
        message = _make_message(db, session, "teaching")
        event = _make_event(
            db, session, through=message.id, count=1,
        )
        with pytest.raises(SessionProfileConflictError):
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

    @pytest.mark.anyio
    async def test_local_reviewer_records_no_remote_ack(self, service, db):
        session = _make_session(db, "local", LOCAL_MODEL)
        message = _make_message(
            db, session, "local claim",
            profile_id="local", kind="local",
            model_snapshot=LOCAL_MODEL,
        )
        event = _make_event(
            db, session, through=message.id, count=1,
        )
        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=True,
            db=db,
        )
        assert result.review.remote_history_acknowledged is False
        assert result.review.remote_history_acknowledged_at is None
        assert result.review.remote_history_ack_message_count is None


class TestRemotePrivacy:
    @pytest.mark.anyio
    async def test_all_same_api_binding_needs_no_ack_and_true_is_not_recorded(
        self, service, db,
    ):
        session = _make_session(db, "api", API_MODEL)
        message = _make_message(
            db, session, "api message",
            profile_id="api", kind="api", model_snapshot=API_MODEL,
        )
        event = _make_event(
            db, session, through=message.id, count=1,
        )

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=True,
            db=db,
        )

        assert result.review.remote_history_acknowledged is False
        assert result.review.remote_history_acknowledged_at is None
        assert result.review.remote_history_ack_message_count is None

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("profile_id", "kind", "model"),
        [
            ("local", "local", LOCAL_MODEL),
            ("default", "fake", FAKE_MODEL),
            (None, None, None),
            ("other-api", "api", API_MODEL),
            ("api", "api", "other-model"),
            ("api", "local", API_MODEL),
        ],
    )
    async def test_api_reviewer_requires_ack_for_non_matching_source(
        self, service, db, profile_id, kind, model,
    ):
        session = _make_session(db, "api", API_MODEL)
        message = _make_message(
            db, session, "source",
            profile_id=profile_id, kind=kind, model_snapshot=model,
        )
        event = _make_event(
            db, session, through=message.id, count=1,
        )

        with pytest.raises(HistoryReviewRemoteAckRequired) as exc:
            await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )

        assert exc.value.source_message_count == 1
        assert exc.value.reviewer_profile_label == "API"
        assert exc.value.reviewer_model == API_MODEL
        assert exc.value.truncated is False
        assert db.execute(
            select(func.count()).select_from(HistoryReview)
        ).scalar() == 0
        assert db.execute(
            select(func.count()).select_from(HistoryReviewSource)
        ).scalar() == 0

    @pytest.mark.anyio
    async def test_acknowledged_source_records_safe_ack_fields(
        self, service, db,
    ):
        session = _make_session(db, "api", API_MODEL)
        message = _make_message(
            db, session, "local source",
            profile_id="local", kind="local", model_snapshot=LOCAL_MODEL,
        )
        event = _make_event(
            db, session, through=message.id, count=1,
        )

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=True,
            db=db,
        )

        assert result.review.remote_history_acknowledged is True
        assert result.review.remote_history_acknowledged_at is not None
        assert result.review.remote_history_ack_message_count == 1


    @pytest.mark.anyio
    async def test_ack_count_matches_selected_source_count_when_truncated(
        self, service, db,
    ):
        session = _make_session(db, "api", API_MODEL)
        messages = [
            _make_message(
                db, session, f"local claim {index}",
                profile_id="local", kind="local",
                model_snapshot=LOCAL_MODEL,
            )
            for index in range(11)
        ]
        event = _make_event(
            db, session, through=messages[-1].id, count=11,
        )

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=True,
            db=db,
        )

        assert result.review.truncated is True
        assert result.review.source_message_count == 10
        assert result.review.remote_history_acknowledged is True
        assert result.review.remote_history_ack_message_count == 10
        assert len(result.sources) == 10

    @pytest.mark.anyio
    async def test_privacy_checks_only_selected_sources_after_truncation(
        self, service, db,
    ):
        session = _make_session(db, "api", API_MODEL)
        old_local = _make_message(
            db, session, "old local claim",
            profile_id="local", kind="local",
            model_snapshot=LOCAL_MODEL,
        )
        api_messages = [
            _make_message(
                db, session, f"api claim {index}",
                profile_id="api", kind="api",
                model_snapshot=API_MODEL,
            )
            for index in range(10)
        ]
        through_message = api_messages[-1]
        event = _make_event(
            db, session, through=through_message.id, count=11,
        )

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=False,
            db=db,
        )

        assert result.created is True
        assert result.review.truncated is True
        assert result.review.source_message_count == 10
        assert result.review.remote_history_acknowledged is False
        assert result.review.remote_history_acknowledged_at is None
        assert result.review.remote_history_ack_message_count is None
        selected_ids = [source.message_id for source in result.sources]
        assert old_local.id not in selected_ids
        assert selected_ids == [message.id for message in api_messages]


def _make_review_row(
    db_session,
    session,
    event,
    *,
    status="pending",
):
    review = HistoryReview(
        session_id=session.id,
        mode_switch_event_id=event.id,
        status=status,
        lower_bound_kind="session_start",
        eligible_message_count=1,
        eligible_char_count=1,
        source_message_count=1,
        source_char_count=1,
        reviewer_llm_profile_id_snapshot="default",
        reviewer_llm_profile_kind_snapshot="fake",
        reviewer_llm_model_snapshot=FAKE_MODEL,
        prompt_version_snapshot=HISTORY_REVIEW_PROMPT_VERSION,
        budget_version_snapshot=HISTORY_REVIEW_BUDGET_VERSION,
        selection_policy_version=HISTORY_REVIEW_SELECTION_POLICY_VERSION,
        attempt_count=1,
    )
    db_session.add(review)
    db_session.commit()
    db_session.refresh(review)
    return review


def _make_source_row(
    db_session, review, message, seq=1,
):
    source = HistoryReviewSource(
        review_id=review.id,
        session_id=review.session_id,
        message_id=message.id,
        seq=seq,
        content_char_count=len(message.content),
    )
    db_session.add(source)
    db_session.commit()
    db_session.refresh(source)
    return source



class TestAtomicityAndIdempotency:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "status",
        ["pending", "running", "completed", "failed"],
    )
    async def test_existing_review_is_returned_without_reexecution(
        self, service, db, status,
    ):
        session = _make_session(db)
        message = _make_message(db, session, "existing claim")
        event = _make_event(
            db, session, through=message.id, count=1,
        )
        review = _make_review_row(db, session, event, status=status)
        source = _make_source_row(db, review, message)
        original_updated_at = review.updated_at
        original_attempt_count = review.attempt_count

        # Make the target boundary and current binding invalid.  The
        # existing-review path must return before either is consulted.
        event.history_boundary_version = None
        session.llm_profile_id = "missing"
        db.commit()

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=False,
            db=db,
        )

        assert result.created is False
        assert result.review.id == review.id
        assert result.review.status == status
        assert [s.id for s in result.sources] == [source.id]
        assert result.review.updated_at == original_updated_at
        assert result.review.attempt_count == original_attempt_count

    @pytest.mark.anyio
    async def test_unique_conflict_converges_to_existing(
        self, service, db,
    ):
        session = _make_session(db)
        message = _make_message(db, session, "existing claim")
        event = _make_event(
            db, session, through=message.id, count=1,
        )
        existing = _make_review_row(db, session, event)
        source = _make_source_row(db, existing, message)

        real_loader = service._load_existing_review
        calls = {"count": 0}

        def fake_loader(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return None
            return real_loader(*args, **kwargs)

        service._load_existing_review = fake_loader
        try:
            result = await service.prepare_history_review(
                session_id=session.id,
                mode_switch_event_id=event.id,
                acknowledge_remote_history=False,
                db=db,
            )
        finally:
            service._load_existing_review = real_loader

        assert calls["count"] >= 2
        assert result.created is False
        assert result.review.id == existing.id
        assert [s.id for s in result.sources] == [source.id]
        assert db.execute(
            select(func.count()).select_from(HistoryReview)
        ).scalar() == 1

    @pytest.mark.anyio
    async def test_source_insert_failure_rolls_back_review(
        self, service, db, engine,
    ):
        session = _make_session(db)
        message = _make_message(db, session, "claim")
        event = _make_event(
            db, session, through=message.id, count=1,
        )
        trigger_name = "fail_history_review_source_insert"
        with engine.begin() as conn:
            conn.execute(sa_text(
                f"CREATE TRIGGER {trigger_name} "
                "BEFORE INSERT ON history_review_sources "
                "BEGIN SELECT RAISE(FAIL, 'simulated source failure'); "
                "END"
            ))
        try:
            with pytest.raises(Exception, match="simulated source failure"):
                await service.prepare_history_review(
                    session_id=session.id,
                    mode_switch_event_id=event.id,
                    acknowledge_remote_history=False,
                    db=db,
                )
        finally:
            with engine.begin() as conn:
                conn.execute(sa_text(
                    f"DROP TRIGGER IF EXISTS {trigger_name}"
                ))

        assert db.execute(
            select(func.count()).select_from(HistoryReview)
        ).scalar() == 0
        assert db.execute(
            select(func.count()).select_from(HistoryReviewSource)
        ).scalar() == 0

    @pytest.mark.anyio
    async def test_stale_identity_map_session_uses_latest_binding(
        self, service, db, session_factory,
    ):
        session = _make_session(db, "default", FAKE_MODEL)
        cached = db.get(ChatSession, session.id)
        assert cached.llm_profile_id == "default"
        db.commit()

        other = session_factory()
        try:
            fresh = other.get(ChatSession, session.id)
            fresh.llm_profile_id = "local"
            fresh.llm_model_snapshot = LOCAL_MODEL
            other.commit()
        finally:
            other.close()

        assert session.llm_profile_id == "default"
        message = _make_message(
            db, session, "local teaching",
            profile_id="local", kind="local", model_snapshot=LOCAL_MODEL,
        )
        event = _make_event(
            db, session, through=message.id, count=1,
        )

        result = await service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=event.id,
            acknowledge_remote_history=False,
            db=db,
        )
        assert result.review.reviewer_llm_profile_id_snapshot == "local"
        assert result.review.reviewer_llm_model_snapshot == LOCAL_MODEL

    @pytest.mark.anyio
    async def test_shared_lock_registry_between_services(
        self, profiles, db,
    ):
        registry = CountingLockRegistry()
        chat_service = ChatService(
            profiles=profiles, lock_registry=registry,
        )
        review_service = HistoryReviewService(
            profiles=profiles, lock_registry=registry,
        )
        session = _make_session(db)
        message = _make_message(db, session, "teaching")
        _make_event(
            db, session, through=None, count=None,
        )  # unrelated event to ensure id ordering unaffected

        _, mode_event = await chat_service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )
        assert mode_event is not None

        result = await review_service.prepare_history_review(
            session_id=session.id,
            mode_switch_event_id=mode_event.id,
            acknowledge_remote_history=False,
            db=db,
        )

        assert result.created is True
        assert message.id <= mode_event.history_through_message_id
        assert registry.entered == [session.id, session.id]

    @pytest.mark.anyio
    async def test_multiple_sessions_do_not_cross(
        self, service, db,
    ):
        session_a = _make_session(db)
        session_b = _make_session(db)
        message_a = _make_message(db, session_a, "a claim")
        message_b = _make_message(db, session_b, "b claim")
        event_a = _make_event(
            db, session_a, through=message_a.id, count=1,
        )
        event_b = _make_event(
            db, session_b, through=message_b.id, count=1,
        )

        result_a = await service.prepare_history_review(
            session_id=session_a.id,
            mode_switch_event_id=event_a.id,
            acknowledge_remote_history=False,
            db=db,
        )
        result_b = await service.prepare_history_review(
            session_id=session_b.id,
            mode_switch_event_id=event_b.id,
            acknowledge_remote_history=False,
            db=db,
        )

        assert result_a.review.session_id == session_a.id
        assert [s.message_id for s in result_a.sources] == [message_a.id]
        assert result_b.review.session_id == session_b.id
        assert [s.message_id for s in result_b.sources] == [message_b.id]
