"""Iteration 2A history-review boundary foundation tests."""

import asyncio
import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy import text as sa_text
from sqlalchemy.orm import sessionmaker

from backend.chat_service import ChatService
from backend.database import (
    create_database_engine,
    create_tables,
    get_db,
    run_migrations,
)
from backend.history_boundary import HISTORY_BOUNDARY_VERSION
from backend.interaction_modes import (
    CORRECTIVE_MODE,
    RECEIVE_TEACHING_MODE,
)
from backend.main import app
from backend.models import ChatSession, Message, ModeSwitchEvent, utc_now

SESSION_PROFILE_MODEL = "injected-test-model"


class DummyLLMClient:
    async def generate(self, messages):
        return "unused"


@pytest.fixture
def test_engine(tmp_path):
    engine = create_database_engine(f"sqlite:///{tmp_path / 'test.db'}")
    create_tables(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def test_session_factory(test_engine):
    return sessionmaker(
        bind=test_engine, autoflush=False, expire_on_commit=False
    )


@pytest.fixture
def db(test_session_factory):
    session = test_session_factory()
    yield session
    session.close()


@pytest.fixture
def service():
    return ChatService(DummyLLMClient())


@pytest.fixture
def client(test_session_factory):
    import backend.main as main_module

    def override_get_db():
        session = test_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    original_service = main_module.chat_service
    main_module.chat_service = ChatService(DummyLLMClient())
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)
        main_module.chat_service = original_service


def _make_session(db_session) -> ChatSession:
    session = ChatSession(llm_model_snapshot=SESSION_PROFILE_MODEL)
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


def _add_message(
    db_session,
    session_id,
    role,
    content,
    interaction_mode_snapshot=None,
) -> Message:
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        interaction_mode_snapshot=interaction_mode_snapshot,
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(message)
    return message


def _event_count(db_session) -> int:
    return db_session.execute(
        select(func.count()).select_from(ModeSwitchEvent)
    ).scalar()


class TestBoundaryCapture:
    @pytest.mark.anyio
    async def test_first_switch_without_messages(self, db, service):
        session = _make_session(db)

        _, event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )

        assert event is not None
        assert event.history_through_message_id is None
        assert event.history_boundary_version == HISTORY_BOUNDARY_VERSION
        assert event.reviewable_user_message_count == 0

    @pytest.mark.anyio
    async def test_through_id_is_max_message_id_before_switch(
        self, db, service,
    ):
        session = _make_session(db)
        first = _add_message(
            db, session.id, "user", "claim one",
            RECEIVE_TEACHING_MODE,
        )
        second = _add_message(
            db, session.id, "user", "claim two",
            RECEIVE_TEACHING_MODE,
        )
        _add_message(
            db, session.id, "assistant", "reply",
            RECEIVE_TEACHING_MODE,
        )
        max_id = db.scalar(select(func.max(Message.id)))

        _, event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )

        assert event.history_through_message_id == max_id
        assert event.history_through_message_id > second.id
        assert event.reviewable_user_message_count == 2
        assert first.id < event.history_through_message_id

    @pytest.mark.anyio
    async def test_new_messages_do_not_change_old_boundary(
        self, db, service,
    ):
        session = _make_session(db)
        _add_message(
            db, session.id, "user", "taught",
            RECEIVE_TEACHING_MODE,
        )
        _, event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )
        old_through = event.history_through_message_id
        old_count = event.reviewable_user_message_count

        _add_message(
            db, session.id, "user", "later message",
            CORRECTIVE_MODE,
        )
        db.refresh(event)

        assert event.history_through_message_id == old_through
        assert event.reviewable_user_message_count == old_count

    @pytest.mark.anyio
    async def test_same_mode_is_noop_without_boundary(self, db, service):
        session = _make_session(db)
        before_updated_at = session.updated_at

        returned, event = await service.switch_interaction_mode(
            session.id, RECEIVE_TEACHING_MODE, db,
        )

        assert event is None
        assert returned.interaction_mode == RECEIVE_TEACHING_MODE
        assert returned.updated_at == before_updated_at
        assert _event_count(db) == 0

    @pytest.mark.anyio
    async def test_two_sessions_parallel_switches_do_not_cross_boundaries(
        self, db, service, test_session_factory,
    ):
        session_a = _make_session(db)
        session_b = _make_session(db)
        _add_message(
            db, session_a.id, "user", "a1", RECEIVE_TEACHING_MODE,
        )
        message_a2 = _add_message(
            db, session_a.id, "user", "a2", RECEIVE_TEACHING_MODE,
        )
        message_b1 = _add_message(
            db, session_b.id, "user", "b1", RECEIVE_TEACHING_MODE,
        )

        async def switch_one(session_id):
            local_db = test_session_factory()
            try:
                _, event = await service.switch_interaction_mode(
                    session_id, CORRECTIVE_MODE, local_db,
                )
                return event
            finally:
                local_db.close()

        event_a, event_b = await asyncio.gather(
            switch_one(session_a.id),
            switch_one(session_b.id),
        )

        assert event_a.history_through_message_id == message_a2.id
        assert event_a.reviewable_user_message_count == 2
        assert event_b.history_through_message_id == message_b1.id
        assert event_b.reviewable_user_message_count == 1

    @pytest.mark.anyio
    async def test_two_sessions_do_not_cross_boundaries(self, db, service):
        session_a = _make_session(db)
        session_b = _make_session(db)
        _add_message(
            db, session_a.id, "user", "a1", RECEIVE_TEACHING_MODE,
        )
        message_a2 = _add_message(
            db, session_a.id, "user", "a2", RECEIVE_TEACHING_MODE,
        )
        message_b1 = _add_message(
            db, session_b.id, "user", "b1", RECEIVE_TEACHING_MODE,
        )
        _add_message(
            db, session_b.id, "assistant", "b reply",
            RECEIVE_TEACHING_MODE,
        )

        _, event_a = await service.switch_interaction_mode(
            session_a.id, CORRECTIVE_MODE, db,
        )
        _, event_b = await service.switch_interaction_mode(
            session_b.id, CORRECTIVE_MODE, db,
        )

        assert event_a.history_through_message_id == message_a2.id
        assert event_a.reviewable_user_message_count == 2
        assert event_b.history_through_message_id != message_a2.id
        assert event_b.history_through_message_id > message_b1.id
        assert event_b.reviewable_user_message_count == 1


    @pytest.mark.anyio
    async def test_authoritative_read_sees_latest_mode_and_noops(
        self, db, service, test_session_factory,
    ):
        session = _make_session(db)
        cached = db.get(ChatSession, session.id)
        assert cached.interaction_mode == RECEIVE_TEACHING_MODE
        # End the read transaction while keeping the stale instance in
        # the identity map.
        db.commit()

        other = test_session_factory()
        try:
            fresh = other.get(ChatSession, session.id)
            fresh.interaction_mode = CORRECTIVE_MODE
            fresh.updated_at = utc_now()
            other.commit()
            other_updated_at = fresh.updated_at
        finally:
            other.close()

        # The original Session still holds the old cached value.
        assert session.interaction_mode == RECEIVE_TEACHING_MODE

        returned, event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )

        assert event is None
        assert returned.interaction_mode == CORRECTIVE_MODE
        assert returned.updated_at == other_updated_at
        assert _event_count(db) == 0

    @pytest.mark.anyio
    async def test_authoritative_read_event_from_mode_is_database_state(
        self, db, service, test_session_factory,
    ):
        session = _make_session(db)
        cached = db.get(ChatSession, session.id)
        assert cached.interaction_mode == RECEIVE_TEACHING_MODE
        db.commit()

        other = test_session_factory()
        try:
            fresh = other.get(ChatSession, session.id)
            fresh.interaction_mode = CORRECTIVE_MODE
            other.commit()
        finally:
            other.close()

        assert session.interaction_mode == RECEIVE_TEACHING_MODE

        # Requested mode equals the stale cached mode, but the database
        # is already corrective.  The service must treat this as a real
        # corrective -> receive_teaching switch.
        returned, event = await service.switch_interaction_mode(
            session.id, RECEIVE_TEACHING_MODE, db,
        )

        assert event is not None
        assert event.from_mode == CORRECTIVE_MODE
        assert event.to_mode == RECEIVE_TEACHING_MODE
        assert returned.interaction_mode == RECEIVE_TEACHING_MODE
        assert _event_count(db) == 1

class TestReviewableUserMessageCount:
    @pytest.mark.anyio
    async def test_filters_non_eligible_messages(self, db, service):
        session = _make_session(db)
        _add_message(
            db, session.id, "user", "eligible claim",
            RECEIVE_TEACHING_MODE,
        )
        _add_message(
            db, session.id, "assistant", "not a user message",
            RECEIVE_TEACHING_MODE,
        )
        _add_message(
            db, session.id, "user", "corrective mode message",
            CORRECTIVE_MODE,
        )
        _add_message(
            db, session.id, "user", "legacy unknown",
            None,
        )

        _, event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )

        assert event.reviewable_user_message_count == 1

    @pytest.mark.anyio
    async def test_upper_bound_is_inclusive(self, db, service):
        session = _make_session(db)
        _add_message(
            db, session.id, "user", "first",
            RECEIVE_TEACHING_MODE,
        )
        last = _add_message(
            db, session.id, "user", "last",
            RECEIVE_TEACHING_MODE,
        )

        _, event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )

        assert event.history_through_message_id == last.id
        assert event.reviewable_user_message_count == 2

    @pytest.mark.anyio
    async def test_reliable_cycle_without_eligible_messages_is_zero(
        self, db, service,
    ):
        session = _make_session(db)
        _add_message(
            db, session.id, "assistant", "reply",
            RECEIVE_TEACHING_MODE,
        )
        _add_message(
            db, session.id, "user", "off mode", CORRECTIVE_MODE,
        )

        _, event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )

        assert event.history_through_message_id is not None
        assert event.reviewable_user_message_count == 0

    @pytest.mark.anyio
    async def test_lower_bound_is_exclusive_across_teaching_cycles(
        self, db, service,
    ):
        session = _make_session(db)
        cycle_one = _add_message(
            db, session.id, "user", "cycle one",
            RECEIVE_TEACHING_MODE,
        )
        _, first_event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )
        assert first_event.reviewable_user_message_count == 1

        _, resume_event = await service.switch_interaction_mode(
            session.id, RECEIVE_TEACHING_MODE, db,
        )
        assert resume_event.history_boundary_version == HISTORY_BOUNDARY_VERSION
        assert resume_event.reviewable_user_message_count is None

        _add_message(
            db, session.id, "user", "cycle two",
            RECEIVE_TEACHING_MODE,
        )
        corrective_user = _add_message(
            db, session.id, "user", "not teaching",
            CORRECTIVE_MODE,
        )

        _, second_event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )

        assert second_event.history_through_message_id == corrective_user.id
        assert second_event.reviewable_user_message_count == 1
        assert cycle_one.id <= resume_event.history_through_message_id

    @pytest.mark.anyio
    async def test_previous_resume_with_legacy_version_makes_count_null(
        self, db, service,
    ):
        session = _make_session(db)
        _add_message(
            db, session.id, "user", "first cycle",
            RECEIVE_TEACHING_MODE,
        )
        legacy_resume = ModeSwitchEvent(
            session_id=session.id,
            from_mode=CORRECTIVE_MODE,
            to_mode=RECEIVE_TEACHING_MODE,
            history_boundary_version=None,
            history_through_message_id=None,
        )
        db.add(legacy_resume)
        db.commit()
        db.refresh(legacy_resume)
        _add_message(
            db, session.id, "user", "current cycle",
            RECEIVE_TEACHING_MODE,
        )

        _, event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )

        assert event.history_boundary_version == HISTORY_BOUNDARY_VERSION
        assert event.history_through_message_id is not None
        assert event.reviewable_user_message_count is None

    @pytest.mark.anyio
    async def test_previous_resume_v1_with_null_through_uses_session_start(
        self, db, service,
    ):
        session = _make_session(db)
        resume_event = ModeSwitchEvent(
            session_id=session.id,
            from_mode=CORRECTIVE_MODE,
            to_mode=RECEIVE_TEACHING_MODE,
            history_boundary_version=HISTORY_BOUNDARY_VERSION,
            history_through_message_id=None,
        )
        db.add(resume_event)
        db.commit()
        _add_message(
            db, session.id, "user", "first after resume",
            RECEIVE_TEACHING_MODE,
        )
        _add_message(
            db, session.id, "user", "second after resume",
            RECEIVE_TEACHING_MODE,
        )

        _, event = await service.switch_interaction_mode(
            session.id, CORRECTIVE_MODE, db,
        )

        assert event.reviewable_user_message_count == 2


class TestModeSwitchApiBoundaryFields:
    def test_empty_session_response_fields(self, client):
        created = client.post("/api/sessions")
        assert created.status_code == 201
        session_id = created.json()["id"]

        switched = client.patch(
            f"/api/sessions/{session_id}/interaction-mode",
            json={"interaction_mode": CORRECTIVE_MODE},
        )
        assert switched.status_code == 200
        event = switched.json()["switch_event"]
        assert event["history_through_message_id"] is None
        assert event["history_boundary_version"] == HISTORY_BOUNDARY_VERSION
        assert event["reviewable_user_message_count"] == 0

    def test_response_fields_with_messages(
        self, client, test_session_factory,
    ):
        created = client.post("/api/sessions")
        session_id = created.json()["id"]

        db = test_session_factory()
        try:
            db.add(Message(
                session_id=session_id,
                role="user",
                content="taught",
                interaction_mode_snapshot=RECEIVE_TEACHING_MODE,
            ))
            db.add(Message(
                session_id=session_id,
                role="assistant",
                content="reply",
                interaction_mode_snapshot=RECEIVE_TEACHING_MODE,
            ))
            db.commit()
            max_id = db.scalar(select(func.max(Message.id)))
        finally:
            db.close()

        switched = client.patch(
            f"/api/sessions/{session_id}/interaction-mode",
            json={"interaction_mode": CORRECTIVE_MODE},
        )
        assert switched.status_code == 200
        event = switched.json()["switch_event"]
        assert event["history_through_message_id"] == max_id
        assert event["history_boundary_version"] == HISTORY_BOUNDARY_VERSION
        assert event["reviewable_user_message_count"] == 1


class TestHistoryBoundaryMigration:
    _BOUNDARY_COLS = {
        "history_through_message_id": "INTEGER",
        "history_boundary_version": "VARCHAR(50)",
        "reviewable_user_message_count": "INTEGER",
    }

    _SKIPPED_MIGRATIONS = (
        "title_is_manual_v1",
        "llm_profile_v1",
        "message_llm_snapshot_v1",
        "interaction_mode_v1",
        "interaction_mode_constraints_v1",
    )

    def _make_legacy_db(self, db_path, present_cols=()):
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0,"
            "  llm_profile_id VARCHAR(50) NOT NULL DEFAULT 'default',"
            "  llm_model_snapshot VARCHAR(255),"
            "  interaction_mode VARCHAR(30) NOT NULL "
            "DEFAULT 'receive_teaching'"
            ")"
        )
        raw.execute(
            "CREATE TABLE messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id INTEGER NOT NULL REFERENCES chat_sessions(id) "
            "ON DELETE CASCADE,"
            "  role VARCHAR(20) NOT NULL,"
            "  content TEXT NOT NULL,"
            "  created_at DATETIME NOT NULL,"
            "  llm_profile_id_snapshot VARCHAR(50),"
            "  llm_profile_kind_snapshot VARCHAR(20),"
            "  llm_model_snapshot VARCHAR(255),"
            "  interaction_mode_snapshot VARCHAR(30),"
            "  prompt_version_snapshot VARCHAR(50)"
            ")"
        )
        raw.execute(
            "CREATE TABLE mode_switch_events ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id INTEGER NOT NULL "
            "REFERENCES chat_sessions(id) ON DELETE CASCADE,"
            "  from_mode VARCHAR(30) NOT NULL,"
            "  to_mode VARCHAR(30) NOT NULL,"
            "  created_at DATETIME NOT NULL"
            ")"
        )
        for col in present_cols:
            raw.execute(
                f"ALTER TABLE mode_switch_events "
                f"ADD COLUMN {col} {self._BOUNDARY_COLS[col]}"
            )
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        for version in self._SKIPPED_MIGRATIONS:
            raw.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)",
                (version,),
            )
        raw.execute(
            "INSERT INTO chat_sessions "
            "(title, created_at, updated_at) "
            "VALUES ('Legacy', '2026-01-01', '2026-01-01')"
        )
        raw.execute(
            "INSERT INTO mode_switch_events "
            "(session_id, from_mode, to_mode, created_at) "
            "VALUES (1, 'receive_teaching', 'corrective', '2026-01-02')"
        )
        raw.commit()
        raw.close()

    def _cols(self, conn, table):
        return [
            row[1]
            for row in conn.execute(
                sa_text(f"PRAGMA table_info('{table}')")
            ).fetchall()
        ]

    def test_new_database_has_boundary_columns(self, test_engine):
        with test_engine.begin() as conn:
            cols = self._cols(conn, "mode_switch_events")
            for col in self._BOUNDARY_COLS:
                assert col in cols

    def test_legacy_database_upgrade_preserves_old_event(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        self._make_legacy_db(db_path)

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(engine)
            with engine.begin() as conn:
                cols = self._cols(conn, "mode_switch_events")
                for col in self._BOUNDARY_COLS:
                    assert cols.count(col) == 1
                row = conn.execute(
                    sa_text(
                        "SELECT history_through_message_id, "
                        "history_boundary_version, "
                        "reviewable_user_message_count "
                        "FROM mode_switch_events WHERE id = 1"
                    )
                ).fetchone()
                assert row == (None, None, None)
                record = conn.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM schema_migrations "
                        "WHERE version = 'history_review_boundary_v1'"
                    )
                ).scalar()
                assert record == 1
        finally:
            engine.dispose()

    def test_partial_columns_converge(self, tmp_path):
        db_path = tmp_path / "partial.db"
        self._make_legacy_db(
            db_path,
            present_cols=(
                "history_through_message_id",
                "reviewable_user_message_count",
            ),
        )

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(engine)
            with engine.begin() as conn:
                cols = self._cols(conn, "mode_switch_events")
                for col in self._BOUNDARY_COLS:
                    assert cols.count(col) == 1
        finally:
            engine.dispose()

    def test_migration_is_idempotent(self, tmp_path):
        db_path = tmp_path / "idempotent.db"
        self._make_legacy_db(db_path)

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(engine)
            run_migrations(engine)
            with engine.begin() as conn:
                count = conn.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM schema_migrations "
                        "WHERE version = 'history_review_boundary_v1'"
                    )
                ).scalar()
                assert count == 1
                cols = self._cols(conn, "mode_switch_events")
                for col in self._BOUNDARY_COLS:
                    assert cols.count(col) == 1
        finally:
            engine.dispose()

    def test_record_insert_failure_then_retry_converges(self, tmp_path):
        db_path = tmp_path / "failure.db"
        self._make_legacy_db(db_path)

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "CREATE TRIGGER fail_boundary_migration_record "
            "BEFORE INSERT ON schema_migrations "
            "WHEN NEW.version = 'history_review_boundary_v1' "
            "BEGIN SELECT RAISE(FAIL, 'simulated failure'); END"
        )
        raw.commit()
        raw.close()

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            with pytest.raises(Exception, match="simulated failure"):
                run_migrations(engine)

            raw2 = sqlite3.connect(str(db_path))
            raw2.execute(
                "DROP TRIGGER IF EXISTS fail_boundary_migration_record"
            )
            raw2.commit()
            raw2.close()

            run_migrations(engine)
            with engine.begin() as conn:
                assert conn.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM schema_migrations "
                        "WHERE version = 'history_review_boundary_v1'"
                    )
                ).scalar() == 1
                cols = self._cols(conn, "mode_switch_events")
                for col in self._BOUNDARY_COLS:
                    assert cols.count(col) == 1
        finally:
            engine.dispose()

    def test_negative_count_rejected_on_migrated_old_db(self, tmp_path):
        db_path = tmp_path / "negative-old.db"
        self._make_legacy_db(db_path)

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(engine)
        finally:
            engine.dispose()

        raw = sqlite3.connect(str(db_path))
        try:
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(
                    "INSERT INTO mode_switch_events "
                    "(session_id, from_mode, to_mode, created_at, "
                    "history_boundary_version, "
                    "reviewable_user_message_count) "
                    "VALUES (1, 'receive_teaching', 'corrective', "
                    "'2026-01-03', 'history-boundary-v1', -1)"
                )
            raw.execute(
                "INSERT INTO mode_switch_events "
                "(session_id, from_mode, to_mode, created_at, "
                "history_boundary_version, reviewable_user_message_count) "
                "VALUES (1, 'receive_teaching', 'corrective', "
                "'2026-01-03', 'history-boundary-v1', NULL)"
            )
            raw.commit()
        finally:
            raw.close()

    def test_negative_count_rejected_on_fresh_db(self, test_engine):
        with test_engine.begin() as conn:
            conn.execute(
                sa_text(
                    "INSERT INTO chat_sessions "
                    "(title, created_at, updated_at) "
                    "VALUES ('Session', '2026-01-03', '2026-01-03')"
                )
            )
            with pytest.raises(Exception):
                conn.execute(
                    sa_text(
                        "INSERT INTO mode_switch_events "
                        "(session_id, from_mode, to_mode, created_at, "
                        "history_boundary_version, "
                        "reviewable_user_message_count) "
                        "VALUES (1, 'receive_teaching', 'corrective', "
                        "'2026-01-03', 'history-boundary-v1', -1)"
                    )
                )


class TestBoundaryRollback:
    @pytest.mark.anyio
    async def test_event_insert_failure_rolls_back_session_and_boundary(
        self, db, service, test_engine, test_session_factory,
    ):
        session = _make_session(db)
        original_updated_at = session.updated_at
        trigger_name = f"fail_mode_switch_event_insert_{session.id}"

        with test_engine.begin() as conn:
            conn.execute(
                sa_text(
                    f"CREATE TRIGGER {trigger_name} "
                    f"BEFORE INSERT ON mode_switch_events "
                    f"WHEN NEW.session_id = {session.id} "
                    "BEGIN "
                    "  SELECT RAISE(FAIL, 'simulated event flush failure'); "
                    "END"
                )
            )

        try:
            with pytest.raises(Exception, match="simulated event flush failure"):
                await service.switch_interaction_mode(
                    session.id, CORRECTIVE_MODE, db,
                )
        finally:
            with test_engine.begin() as conn:
                conn.execute(
                    sa_text(f"DROP TRIGGER IF EXISTS {trigger_name}")
                )

        fresh = test_session_factory()
        try:
            loaded = fresh.get(ChatSession, session.id)
            assert loaded.interaction_mode == RECEIVE_TEACHING_MODE
            assert loaded.updated_at == original_updated_at
            event_count = fresh.execute(
                select(func.count()).select_from(ModeSwitchEvent)
            ).scalar()
            assert event_count == 0
        finally:
            fresh.close()
