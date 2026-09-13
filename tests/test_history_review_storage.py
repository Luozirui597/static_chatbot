"""Iteration 2B storage schema, ORM, and constraint tests."""

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.database import create_database_engine, create_tables
from backend.interaction_modes import CORRECTIVE_MODE, RECEIVE_TEACHING_MODE
from backend.models import (
    ChatSession,
    HistoryReview,
    HistoryReviewFinding,
    HistoryReviewSource,
    Message,
    ModeSwitchEvent,
)


@pytest.fixture
def engine(tmp_path):
    eng = create_database_engine(f"sqlite:///{tmp_path / 'storage.db'}")
    create_tables(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    factory = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False
    )
    session = factory()
    yield session
    session.close()


def _make_session(db_session, title="Session") -> ChatSession:
    session = ChatSession(
        title=title, llm_model_snapshot="injected-test-model"
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


def _make_event(
    db_session, session: ChatSession, from_mode=RECEIVE_TEACHING_MODE,
) -> ModeSwitchEvent:
    event = ModeSwitchEvent(
        session_id=session.id,
        from_mode=from_mode,
        to_mode=CORRECTIVE_MODE,
        history_boundary_version="history-boundary-v1",
    )
    db_session.add(event)
    db_session.commit()
    db_session.refresh(event)
    return event


def _make_message(
    db_session, session: ChatSession, content="hello",
) -> Message:
    message = Message(
        session_id=session.id,
        role="user",
        content=content,
        interaction_mode_snapshot=RECEIVE_TEACHING_MODE,
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(message)
    return message


def _make_review(
    db_session, session: ChatSession, event: ModeSwitchEvent,
) -> HistoryReview:
    review = HistoryReview(
        session_id=session.id,
        mode_switch_event_id=event.id,
        status="pending",
        lower_bound_kind="session_start",
        eligible_message_count=1,
        eligible_char_count=5,
        source_message_count=1,
        source_char_count=5,
        reviewer_llm_profile_id_snapshot="default",
        reviewer_llm_profile_kind_snapshot="fake",
        reviewer_llm_model_snapshot="fake",
        prompt_version_snapshot="history-review-v1",
        budget_version_snapshot="history-review-budget-v1",
        selection_policy_version="history-review-selection-v1",
    )
    db_session.add(review)
    db_session.commit()
    db_session.refresh(review)
    return review


def _make_source(
    db_session, review: HistoryReview, message: Message, seq=1,
) -> HistoryReviewSource:
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


def _make_finding(
    db_session, review: HistoryReview, source: HistoryReviewSource,
    seq=1,
) -> HistoryReviewFinding:
    finding = HistoryReviewFinding(
        review_id=review.id,
        seq=seq,
        verdict="correct",
        claim_text="claim",
        source_message_id=source.message_id,
    )
    db_session.add(finding)
    db_session.commit()
    db_session.refresh(finding)
    return finding


class TestFreshSchema:
    def test_tables_and_columns_exist(self, engine):
        inspector = inspect(engine)
        assert {
            "history_reviews",
            "history_review_sources",
            "history_review_findings",
        }.issubset(set(inspector.get_table_names()))

        review_columns = {
            col["name"] for col in inspector.get_columns("history_reviews")
        }
        assert {
            "id",
            "session_id",
            "mode_switch_event_id",
            "status",
            "lower_bound_kind",
            "eligible_message_count",
            "source_message_count",
            "truncated",
            "reviewer_llm_profile_id_snapshot",
            "budget_version_snapshot",
            "selection_policy_version",
            "remote_history_ack_message_count",
        }.issubset(review_columns)

        source_columns = {
            col["name"]
            for col in inspector.get_columns("history_review_sources")
        }
        assert {
            "id",
            "review_id",
            "session_id",
            "message_id",
            "seq",
            "content_char_count",
        }.issubset(source_columns)

        finding_columns = {
            col["name"]
            for col in inspector.get_columns("history_review_findings")
        }
        assert {
            "id",
            "review_id",
            "seq",
            "verdict",
            "claim_text",
            "source_message_id",
        }.issubset(finding_columns)

    def test_parent_unique_indexes_exist_for_composite_fks(self, engine):
        inspector = inspect(engine)
        messages_uniques = inspector.get_unique_constraints("messages")
        events_uniques = inspector.get_unique_constraints(
            "mode_switch_events"
        )
        assert any(
            set(unique["column_names"]) == {"id", "session_id"}
            for unique in messages_uniques
        )
        assert any(
            set(unique["column_names"]) == {"id", "session_id"}
            for unique in events_uniques
        )

    def test_valid_review_source_finding_roundtrip(self, db, engine):
        session = _make_session(db)
        event = _make_event(db, session)
        message = _make_message(db, session)
        review = _make_review(db, session, event)
        source = _make_source(db, review, message)
        finding = _make_finding(db, review, source)

        assert review.id is not None
        assert source.review_id == review.id
        assert source.message_id == message.id
        assert finding.review_id == review.id
        assert finding.source_message_id == message.id
        with engine.begin() as conn:
            assert conn.execute(
                sa_text("PRAGMA foreign_key_check")
            ).fetchall() == []

    def test_one_review_per_session_event(self, db):
        session = _make_session(db)
        event = _make_event(db, session)
        _make_review(db, session, event)

        duplicate = HistoryReview(
            session_id=session.id,
            mode_switch_event_id=event.id,
            status="pending",
            lower_bound_kind="session_start",
            eligible_message_count=1,
            eligible_char_count=5,
            source_message_count=1,
            source_char_count=5,
            reviewer_llm_profile_id_snapshot="default",
            reviewer_llm_profile_kind_snapshot="fake",
            reviewer_llm_model_snapshot="fake",
            prompt_version_snapshot="history-review-v1",
            budget_version_snapshot="history-review-budget-v1",
            selection_policy_version="history-review-selection-v1",
        )
        db.add(duplicate)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_review_cannot_reference_other_session_event(self, db):
        session_a = _make_session(db, "A")
        session_b = _make_session(db, "B")
        event_b = _make_event(db, session_b)

        review = HistoryReview(
            session_id=session_a.id,
            mode_switch_event_id=event_b.id,
            status="pending",
            lower_bound_kind="session_start",
            eligible_message_count=1,
            eligible_char_count=5,
            source_message_count=1,
            source_char_count=5,
            reviewer_llm_profile_id_snapshot="default",
            reviewer_llm_profile_kind_snapshot="fake",
            reviewer_llm_model_snapshot="fake",
            prompt_version_snapshot="history-review-v1",
            budget_version_snapshot="history-review-budget-v1",
            selection_policy_version="history-review-selection-v1",
        )
        db.add(review)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_source_cannot_cross_session_message(self, db):
        session_a = _make_session(db, "A")
        session_b = _make_session(db, "B")
        event_a = _make_event(db, session_a)
        review_a = _make_review(db, session_a, event_a)
        message_b = _make_message(db, session_b)

        source = HistoryReviewSource(
            review_id=review_a.id,
            session_id=session_a.id,
            message_id=message_b.id,
            seq=1,
            content_char_count=len(message_b.content),
        )
        db.add(source)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_source_cannot_forge_review_session(self, db):
        session_a = _make_session(db, "A")
        session_b = _make_session(db, "B")
        event_a = _make_event(db, session_a)
        review_a = _make_review(db, session_a, event_a)
        message_b = _make_message(db, session_b)

        source = HistoryReviewSource(
            review_id=review_a.id,
            session_id=session_b.id,
            message_id=message_b.id,
            seq=1,
            content_char_count=len(message_b.content),
        )
        db.add(source)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_finding_cannot_reference_other_review_source(self, db):
        session = _make_session(db)
        event_a = _make_event(db, session)
        event_b = _make_event(db, session)
        message_a = _make_message(db, session, "first")
        message_b = _make_message(db, session, "second")

        review_a = _make_review(db, session, event_a)
        review_b = _make_review(db, session, event_b)
        source_a = _make_source(db, review_a, message_a, seq=1)
        source_b = _make_source(db, review_b, message_b, seq=1)

        assert source_a.message_id == message_a.id
        assert source_b.message_id == message_b.id

        cross_finding = HistoryReviewFinding(
            review_id=review_a.id,
            seq=1,
            verdict="correct",
            claim_text="cross review source",
            source_message_id=source_b.message_id,
        )
        db.add(cross_finding)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_source_unique_constraints(self, db):
        session = _make_session(db)
        event = _make_event(db, session)
        message = _make_message(db, session)
        second_message = _make_message(db, session, "second")
        review = _make_review(db, session, event)
        _make_source(db, review, message, seq=1)

        duplicate = HistoryReviewSource(
            review_id=review.id,
            session_id=session.id,
            message_id=message.id,
            seq=2,
            content_char_count=len(message.content),
        )
        db.add(duplicate)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        duplicate_seq = HistoryReviewSource(
            review_id=review.id,
            session_id=session.id,
            message_id=second_message.id,
            seq=1,
            content_char_count=len(second_message.content),
        )
        db.add(duplicate_seq)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_review_checks_reject_invalid_values(self, db, engine):
        session = _make_session(db)
        event = _make_event(db, session)
        review = _make_review(db, session, event)

        invalid_updates = [
            "UPDATE history_reviews SET status = 'bogus' WHERE id = :id",
            "UPDATE history_reviews SET lower_bound_kind = 'bogus' "
            "WHERE id = :id",
            "UPDATE history_reviews SET "
            "reviewer_llm_profile_kind_snapshot = 'bogus' WHERE id = :id",
            "UPDATE history_reviews SET eligible_message_count = 0 "
            "WHERE id = :id",
            "UPDATE history_reviews SET source_message_count = 0 "
            "WHERE id = :id",
            "UPDATE history_reviews SET eligible_message_count = 0, "
            "source_message_count = 1 WHERE id = :id",
            "UPDATE history_reviews SET eligible_char_count = -1 "
            "WHERE id = :id",
            "UPDATE history_reviews SET source_char_count = -1 "
            "WHERE id = :id",
            "UPDATE history_reviews SET attempt_count = 0 WHERE id = :id",
            "UPDATE history_reviews SET "
            "remote_history_ack_message_count = -1 WHERE id = :id",
            "UPDATE history_reviews SET truncated = 2 WHERE id = :id",
            "UPDATE history_reviews SET raw_output_truncated = 2 "
            "WHERE id = :id",
            "UPDATE history_reviews SET remote_history_acknowledged = 2 "
            "WHERE id = :id",
        ]
        for statement in invalid_updates:
            with pytest.raises(IntegrityError):
                with engine.begin() as conn:
                    conn.execute(sa_text(statement), {"id": review.id})

    def test_source_and_finding_checks_reject_invalid_values(self, db, engine):
        session = _make_session(db)
        event = _make_event(db, session)
        message = _make_message(db, session)
        review = _make_review(db, session, event)
        source = _make_source(db, review, message)
        finding = _make_finding(db, review, source)

        invalid_updates = [
            ("UPDATE history_review_sources SET seq = 0 WHERE id = :id",
             {"id": source.id}),
            ("UPDATE history_review_sources "
             "SET content_char_count = -1 WHERE id = :id",
             {"id": source.id}),
            ("UPDATE history_review_findings SET seq = 0 WHERE id = :id",
             {"id": finding.id}),
            ("UPDATE history_review_findings SET verdict = 'bogus' "
             "WHERE id = :id", {"id": finding.id}),
            ("UPDATE history_review_findings SET claim_text = '   ' "
             "WHERE id = :id", {"id": finding.id}),
        ]
        for statement, params in invalid_updates:
            with pytest.raises(IntegrityError):
                with engine.begin() as conn:
                    conn.execute(sa_text(statement), params)

    def test_session_delete_cascades_reviews_sources_findings(self, db):
        session = _make_session(db)
        event = _make_event(db, session)
        message = _make_message(db, session)
        review = _make_review(db, session, event)
        source = _make_source(db, review, message)
        _make_finding(db, review, source)

        db.delete(session)
        db.commit()
        db.expire_all()

        assert db.execute(
            select(func.count()).select_from(HistoryReview)
        ).scalar() == 0
        assert db.execute(
            select(func.count()).select_from(HistoryReviewSource)
        ).scalar() == 0
        assert db.execute(
            select(func.count()).select_from(HistoryReviewFinding)
        ).scalar() == 0

    def test_review_delete_cascades_sources_findings(self, db):
        session = _make_session(db)
        event = _make_event(db, session)
        message = _make_message(db, session)
        review = _make_review(db, session, event)
        source = _make_source(db, review, message)
        _make_finding(db, review, source)

        db.delete(review)
        db.commit()

        assert db.execute(
            select(func.count()).select_from(HistoryReviewSource)
        ).scalar() == 0
        assert db.execute(
            select(func.count()).select_from(HistoryReviewFinding)
        ).scalar() == 0


class TestEventCascade:
    def test_event_delete_cascades_review_sources_findings(self, db):
        session = _make_session(db)
        event = _make_event(db, session)
        message = _make_message(db, session)
        review = _make_review(db, session, event)
        source = _make_source(db, review, message)
        _make_finding(db, review, source)

        db.delete(event)
        db.commit()

        assert db.execute(
            select(func.count()).select_from(HistoryReview)
        ).scalar() == 0
        assert db.execute(
            select(func.count()).select_from(HistoryReviewSource)
        ).scalar() == 0
        assert db.execute(
            select(func.count()).select_from(HistoryReviewFinding)
        ).scalar() == 0
